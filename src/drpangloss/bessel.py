"""Bessel functions of the first kind, J_n(x), in JAX.

Self-contained (it depends only on JAX and NumPy) so that it can move to a
standalone package shared with harmonix and other projects.

``j0`` and ``j1`` follow the [CEPHES](https://www.netlib.org/cephes/)
rational approximations, with the JAX translation adapted from
[Harmonix](https://github.com/shashankdholakia/harmonix). ``bessel_jn``
returns all orders up to ``n``: for ``|x| < n + 2`` it uses a folded
trapezoidal trigonometric sum (arXiv:2206.05334), where upward recurrence is
unstable, and the recurrence above it. Values and gradients are finite
everywhere, including at ``x = 0``.
"""

from functools import partial

import jax.numpy as np
import numpy as onp
from jax import jit
from jax.lax import scan


__all__ = ["bessel_jn", "j0", "j1"]

# === BESSEL FUNCTIONS OF THE FIRST KIND, BASED ON THE CEPHES IMPLEMENTATION ===
#
# The coefficient tables are NumPy arrays, not JAX arrays, so they stay float64
# even when drpangloss is imported before ``jax_enable_x64`` is set.

RP1 = onp.array(
    [
        -8.99971225705559398224e8,
        4.52228297998194034323e11,
        -7.27494245221818276015e13,
        3.68295732863852883286e15,
    ]
)
RQ1 = onp.array(
    [
        1.0,
        6.20836478118054335476e2,
        2.56987256757748830383e5,
        8.35146791431949253037e7,
        2.21511595479792499675e10,
        4.74914122079991414898e12,
        7.84369607876235854894e14,
        8.95222336184627338078e16,
        5.32278620332680085395e18,
    ]
)

PP1 = onp.array(
    [
        7.62125616208173112003e-4,
        7.31397056940917570436e-2,
        1.12719608129684925192e0,
        5.11207951146807644818e0,
        8.42404590141772420927e0,
        5.21451598682361504063e0,
        1.00000000000000000254e0,
    ]
)
PQ1 = onp.array(
    [
        5.71323128072548699714e-4,
        6.88455908754495404082e-2,
        1.10514232634061696926e0,
        5.07386386128601488557e0,
        8.39985554327604159757e0,
        5.20982848682361821619e0,
        9.99999999999999997461e-1,
    ]
)

QP1 = onp.array(
    [
        5.10862594750176621635e-2,
        4.98213872951233449420e0,
        7.58238284132545283818e1,
        3.66779609360150777800e2,
        7.10856304998926107277e2,
        5.97489612400613639965e2,
        2.11688757100572135698e2,
        2.52070205858023719784e1,
    ]
)
QQ1 = onp.array(
    [
        1.0,
        7.42373277035675149943e1,
        1.05644886038262816351e3,
        4.98641058337653607651e3,
        9.56231892404756170795e3,
        7.99704160447350683650e3,
        2.82619278517639096600e3,
        3.36093607810698293419e2,
    ]
)

YP1 = onp.array(
    [
        1.26320474790178026440e9,
        -6.47355876379160291031e11,
        1.14509511541823727583e14,
        -8.12770255501325109621e15,
        2.02439475713594898196e17,
        -7.78877196265950026825e17,
    ]
)
YQ1 = onp.array(
    [
        5.94301592346128195359e2,
        2.35564092943068577943e5,
        7.34811944459721705660e7,
        1.87601316108706159478e10,
        3.88231277496238566008e12,
        6.20557727146953693363e14,
        6.87141087355300489866e16,
        3.97270608116560655612e18,
    ]
)

Z1 = 1.46819706421238932572e1
Z2 = 4.92184563216946036703e1
PIO4 = 0.78539816339744830962  # pi/4
THPIO4 = 2.35619449019234492885  # 3*pi/4
SQ2OPI = 0.79788456080286535588  # sqrt(2/pi)


def j1_small(x):
    z = x * x
    w = np.polyval(RP1, z) / np.polyval(RQ1, z)
    w = w * x * (z - Z1) * (z - Z2)
    return w


def j1_large_c(x):
    w = 5.0 / x
    z = w * w
    p = np.polyval(PP1, z) / np.polyval(PQ1, z)
    q = np.polyval(QP1, z) / np.polyval(QQ1, z)
    xn = x - THPIO4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    return p * SQ2OPI / np.sqrt(x)


def j1(x):
    """Bessel function of order one, translated from the CEPHES implementation."""
    ax = np.abs(x)
    small = ax < 5.0
    # Feed each branch only arguments in its own range, so the unused branch
    # cannot produce NaN gradients (e.g. 5 / x at x = 0). ``j1_small`` is odd,
    # so it takes the signed argument and keeps the gradient at x = 0.
    return np.where(
        small,
        j1_small(np.where(small, x, 0.0)),
        np.sign(x) * j1_large_c(np.where(small, 5.0, ax)),
    )


PP0 = onp.array(
    [
        7.96936729297347051624e-4,
        8.28352392107440799803e-2,
        1.23953371646414299388e0,
        5.44725003058768775090e0,
        8.74716500199817011941e0,
        5.30324038235394892183e0,
        9.99999999999999997821e-1,
    ]
)
PQ0 = onp.array(
    [
        9.24408810558863637013e-4,
        8.56288474354474431428e-2,
        1.25352743901058953537e0,
        5.47097740330417105182e0,
        8.76190883237069594232e0,
        5.30605288235394617618e0,
        1.00000000000000000218e0,
    ]
)

QP0 = onp.array(
    [
        -1.13663838898469149931e-2,
        -1.28252718670509318512e0,
        -1.95539544257735972385e1,
        -9.32060152123768231369e1,
        -1.77681167980488050595e2,
        -1.47077505154951170175e2,
        -5.14105326766599330220e1,
        -6.05014350600728481186e0,
    ]
)
QQ0 = onp.array(
    [
        1.0,
        6.43178256118178023184e1,
        8.56430025976980587198e2,
        3.88240183605401609683e3,
        7.24046774195652478189e3,
        5.93072701187316984827e3,
        2.06209331660327847417e3,
        2.42005740240291393179e2,
    ]
)

YP0 = onp.array(
    [
        1.55924367855235737965e4,
        -1.46639295903971606143e7,
        5.43526477051876500413e9,
        -9.82136065717911466409e11,
        8.75906394395366999549e13,
        -3.46628303384729719441e15,
        4.42733268572569800351e16,
        -1.84950800436986690637e16,
    ]
)
YQ0 = onp.array(
    [
        1.04128353664259848412e3,
        6.26107330137134956842e5,
        2.68919633393814121987e8,
        8.64002487103935000337e10,
        2.02979612750105546709e13,
        3.17157752842975028269e15,
        2.50596256172653059228e17,
    ]
)

DR10 = 5.78318596294678452118e0
DR20 = 3.04712623436620863991e1

RP0 = onp.array(
    [
        -4.79443220978201773821e9,
        1.95617491946556577543e12,
        -2.49248344360967716204e14,
        9.70862251047306323952e15,
    ]
)
RQ0 = onp.array(
    [
        1.0,
        4.99563147152651017219e2,
        1.73785401676374683123e5,
        4.84409658339962045305e7,
        1.11855537045356834862e10,
        2.11277520115489217587e12,
        3.10518229857422583814e14,
        3.18121955943204943306e16,
        1.71086294081043136091e18,
    ]
)


def j0_small(x):
    """Implementation of J0 for x < 5."""
    z = x * x
    p = (z - DR10) * (z - DR20)
    p = p * np.polyval(RP0, z) / np.polyval(RQ0, z)
    return np.where(x < 1e-5, 1 - z / 4.0, p)


def j0_large(x):
    """Implementation of J0 for x >= 5."""
    w = 5.0 / x
    q = 25.0 / (x * x)
    p = np.polyval(PP0, q) / np.polyval(PQ0, q)
    q = np.polyval(QP0, q) / np.polyval(QQ0, q)
    xn = x - PIO4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    return p * SQ2OPI / np.sqrt(x)


def j0(x):
    """Implementation of J0 for all x in Jax."""
    ax = np.abs(x)
    small = ax < 5.0
    return np.where(
        small,
        j0_small(np.where(small, ax, 0.0)),
        j0_large(np.where(small, 5.0, ax)),
    )


def _bessel_jn_trig(n, x, nodes):
    r"""All orders ``0..n`` of $J_m(x)$ from an ``nodes``-point trigonometric sum.

    The trapezoidal rule applied to Bessel's integral
    $J_m(x) = \frac{1}{2\pi}\int_0^{2\pi} \cos(m t - x \sin t)\,dt$
    gives $J_m(x) + \sum_{k \neq 0} (\pm) J_{m + k N}(x)$ (see
    arXiv:2206.05334), so it matches $J_m(x)$ to machine precision while
    $J_{N - m}(x)$ is negligible, i.e. for $|x|$ well below $N - m$.

    ``nodes`` must be a multiple of 4. The nodes' $\sin t$ values repeat, up
    to sign, over the quarter period, so only ``nodes / 4 + 1`` cosines and
    sines of $x$ are needed; the weights are folded together at trace time.
    """
    k = onp.arange(nodes)
    t = 2.0 * onp.pi * k / nodes
    half = k % (nodes // 2)
    quarter = onp.where(half <= nodes // 4, half, nodes // 2 - half)
    sign = onp.where(k < nodes // 2, 1.0, -1.0)
    orders = onp.arange(n + 1)[:, None]
    # cos(m t - x sin t) = cos(m t) cos(x sin t) + sin(m t) sin(x sin t)
    w_cos = onp.zeros((n + 1, nodes // 4 + 1))
    w_sin = onp.zeros((n + 1, nodes // 4 + 1))
    onp.add.at(w_cos.T, quarter, (onp.cos(orders * t) / nodes).T)
    onp.add.at(w_sin.T, quarter, (sign * onp.sin(orders * t) / nodes).T)
    weights = np.asarray(onp.concatenate([w_cos, w_sin], axis=1))

    arg = x[..., None] * np.sin(
        2.0 * np.pi * onp.arange(nodes // 4 + 1) / nodes
    )
    trig = np.concatenate([np.cos(arg), np.sin(arg)], axis=-1)
    return np.moveaxis(trig @ weights.T, -1, 0)


@partial(jit, static_argnums=0)
def bessel_jn(n, x):
    r"""Compute the Bessel function $J_n(x)$, for $n >= 0$. Returns the function output
    for all orders up to the requested order $n$ evaluated for the kernel $x$, where the
    the different orders are stacked along the first axis. The shape of the final result
    is thus (n + 1, shape(x)).

    For $n \le 1$ this is the CEPHES rational approximations. For $n \ge 2$, all
    orders come from the upward recurrence $J_{m+1} = (2m/x) J_m - J_{m-1}$,
    seeded by CEPHES, where it is stable ($|x| > n$), and from a
    trigonometric sum (see ``_bessel_jn_trig``) below that, where the
    recurrence loses accuracy. Both agree with
    ``scipy.special.jv`` to about 1e-14, and the gradients are finite
    everywhere, including $x = 0$.
    """
    x = np.asarray(x, dtype=float)
    if n == 0:
        return j0(x)[None]
    if n == 1:
        return np.stack([j0(x), j1(x)])

    # The recurrence is accurate for |x| > n. With 3n + 24 nodes (rounded up
    # to a multiple of 4) the trigonometric sum is accurate to machine
    # precision up to x_switch for every order <= n.
    x_switch = float(n + 2)
    nodes = 4 * -(-(3 * n + 24) // 4)
    small = np.abs(x) < x_switch
    x_rec = np.where(small, x_switch, x)

    def body(carry, i):
        jnm1, jn = carry
        jnplus = (2 * i) / x_rec * jn - jnm1
        return (jn, jnplus), jnplus

    j0_rec, j1_rec = j0(x_rec), j1(x_rec)
    _, j_high = scan(body, (j0_rec, j1_rec), np.arange(1, n))
    j_rec = np.concatenate([j0_rec[None], j1_rec[None], j_high])
    j_trig = _bessel_jn_trig(n, np.where(small, x, 0.0), nodes)
    return np.where(small, j_trig, j_rec)
