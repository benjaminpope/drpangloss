"""Deprecation helpers for the 0.4 argument-order change (private).

virgil 0.4 puts the model before the data in every public function that
takes both, renames ``samples_dict`` to ``grid``, and standardizes the
argument names ``model`` and ``data``. :func:`old_order` lets calls in
the 0.3 positional order keep working with a :class:`FutureWarning`, and
:func:`renamed` accepts the old keyword names. In 0.5 the aliases go and
``old_order(..., removed=True)`` turns the old order into a
:class:`TypeError` naming the new call.

The checks look only at the Python types of the arguments, never at
array values, so they are valid while JAX traces a call.
"""

import functools
import inspect
import sys
import warnings

__all__ = ["old_order", "renamed"]

#: Old keyword names accepted until 0.5, mapped to their 0.4 names.
NAME_ALIASES = {
    "data_obj": "data",
    "observations": "data",
    "model_object": "model",
    "model_fn": "model",
    "samples_dict": "grid",
    "ud": "diam",
    # Oxford -ize spelling (0.4)
    "regularisers": "regularizers",
    "marginalise_offsets": "marginalize_offsets",
    "regulariser": "regularizer",
}

#: Top-level packages whose frames a deprecation warning looks through.
_INTERNAL = (
    "virgil",
    "equinox",
    "functools",
    "jax",
    "jaxlib",
    "jaxtyping",
    "lineax",
    "numpyro",
    "optax",
    "optimistix",
)


def external_stacklevel():
    """The ``stacklevel`` of the first caller outside virgil and its wrappers.

    Meant for ``warnings.warn(..., stacklevel=external_stacklevel())`` in
    the function that warns, so the warning is attributed to user code
    even when equinox's method wrappers, ``functools``, jaxtyping, or the
    tracing of JAX, numpyro, optimistix, lineax or optax add frames
    between the two. Falls back to the outermost frame.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        top = frame.f_globals.get("__name__", "").partition(".")[0]
        if top not in _INTERNAL:
            return level
        frame = frame.f_back
        level += 1
    return level - 1


def _is_data(x):
    """Whether ``x`` is an ``OIData`` or a non-empty sequence of them."""
    from .oidata import OIData  # lazy: keeps this module import-free

    if isinstance(x, OIData):
        return True
    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(isinstance(d, OIData) for d in x)
    )


def _alias_map(params, aliases):
    """The aliases in ``aliases`` whose new name is a parameter of ``fn``."""
    return {
        old: new
        for old, new in aliases.items()
        if new in params and old not in params
    }


def _apply_aliases(name, args, kwargs, aliases, positional=()):
    """Rename deprecated keywords in ``kwargs`` in place, with a warning.

    ``positional`` lists the parameter names in positional order, to catch
    a new name passed positionally beside its old keyword.
    """
    for old, new in aliases.items():
        if old not in kwargs:
            continue
        if new in kwargs or (
            new in positional and positional.index(new) < len(args)
        ):
            raise TypeError(f"{name} got both {new}= and {old}=; use {new}=")
        warnings.warn(
            f"{name}({old}=...) is now {name}({new}=...); "
            f"{old}= stops working in virgil 0.5.",
            FutureWarning,
            stacklevel=external_stacklevel(),
        )
        kwargs[new] = kwargs.pop(old)


def _positional_names(params):
    """Names of the parameters that can be passed positionally, in order."""
    kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    return tuple(n for n, p in params.items() if p.kind in kinds)


def _as_keywords(bound, params):
    """Flatten a binding to keywords, expanding any ``**kwargs``."""
    out = {}
    for name, value in bound.arguments.items():
        if params[name].kind is inspect.Parameter.VAR_KEYWORD:
            out.update(value)
        else:
            out[name] = value
    return out


def renamed(aliases=None):
    """Accept the pre-0.4 keyword names of ``fn``'s parameters until 0.5.

    ``aliases`` maps old names to new ones; it defaults to
    :data:`NAME_ALIASES`, restricted to the new names ``fn`` has.
    """

    def decorate(fn):
        params = inspect.signature(fn).parameters
        names = _alias_map(
            params, NAME_ALIASES if aliases is None else aliases
        )
        positional = _positional_names(params)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _apply_aliases(fn.__qualname__, args, kwargs, names, positional)
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def old_order(*old_names, data, aliases=None, removed=False):
    """Accept calls in the pre-0.4 positional order, with a FutureWarning.

    Parameters
    ----------
    *old_names : str
        The leading positional parameters in their old order, under their
        new names.
    data : str
        The name of the data parameter, which is how an old-order call is
        recognized: the argument at its old position is data and the one
        at its new position is not.
    aliases : dict, optional
        Old keyword names mapped to new ones, as for :func:`renamed`.
    removed : bool
        Raise a :class:`TypeError` naming the new call instead of
        rebinding (the 0.5 behaviour).

    Calls with data at both positions or at neither are passed through
    unchanged, so new-order code never warns.
    """

    def decorate(fn):
        sig = inspect.signature(fn)
        params = sig.parameters
        unknown = [n for n in (*old_names, data) if n not in params]
        if unknown:
            raise ValueError(f"{fn.__name__} has no parameters {unknown}")
        names = _alias_map(
            params, NAME_ALIASES if aliases is None else aliases
        )
        positional = _positional_names(params)
        qualname = fn.__qualname__
        new_names = [n for n in params if n in old_names]
        moved = [params[n] for n in old_names] + [
            p for n, p in params.items() if n not in old_names
        ]
        old_sig = sig.replace(parameters=moved)
        i_old, i_new = old_names.index(data), new_names.index(data)
        new_call = f"{qualname}({', '.join(new_names)}, ...)"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _apply_aliases(qualname, args, kwargs, names, positional)
            if not (
                len(args) > i_old
                and _is_data(args[i_old])
                and not (len(args) > i_new and _is_data(args[i_new]))
            ):
                return fn(*args, **kwargs)
            if removed:
                raise TypeError(
                    f"{qualname} takes the model before the data since "
                    f"virgil 0.4; call {new_call}."
                )
            warnings.warn(
                f"{qualname} now takes the model before the data; "
                f"call {new_call}. The old order stops working in "
                "virgil 0.5.",
                FutureWarning,
                stacklevel=external_stacklevel(),
            )
            bound = old_sig.bind(*args, **kwargs)
            return fn(**_as_keywords(bound, params))

        return wrapper

    return decorate
