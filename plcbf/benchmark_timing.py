"""Steady-state safety-filter timing samples."""

from threading import local

from jax import monitoring

_compilation_state = local()


def _compilation_finished(event, duration, **metadata):
    if event == "/jax/core/compile/backend_compile_duration":
        _compilation_state.count = getattr(_compilation_state, "count", 0) + 1


monitoring.register_event_duration_secs_listener(_compilation_finished)


class BenchmarkTiming:
    def __init__(self, warmup_steps=10, compilation_cooldown=3):
        self.warmup_steps = warmup_steps
        self.compilation_cooldown = compilation_cooldown
        self.samples = []
        self.steps = 0
        self.last_compilation = -compilation_cooldown - 1
        self.compilations = getattr(_compilation_state, "count", 0)

    def record(self, elapsed):
        count = getattr(_compilation_state, "count", 0)
        if count != self.compilations:
            self.last_compilation = self.steps
        self.compilations = count
        retain = (
            self.steps >= self.warmup_steps
            and self.steps > self.last_compilation + self.compilation_cooldown
        )
        self.steps += 1
        if retain:
            self.samples.append(float(elapsed))
        return retain
