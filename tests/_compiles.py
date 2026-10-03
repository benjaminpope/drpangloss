import contextlib

import jax


@contextlib.contextmanager
def count_compiles():
    """Count XLA backend compiles inside the block, into the yielded list."""
    compiles = []

    def count(event, duration, **kwargs):
        if event.endswith("backend_compile_duration"):
            compiles.append(event)

    jax.monitoring.register_event_duration_secs_listener(count)
    try:
        yield compiles
    finally:
        # The listener is process-global. Older JAX only has the private
        # helper.
        unregister = getattr(
            jax.monitoring,
            "unregister_event_duration_listener",
            None,
        ) or getattr(
            jax._src.monitoring,
            "_unregister_event_duration_listener_by_callback",
        )
        unregister(count)
