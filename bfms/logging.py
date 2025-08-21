import jax


class RecordWriter:
    def __init__(self):
        self._prev_metrics = None
        self._prev_step = None

    def __call__(self, metrics: dict, step: int) -> None:
        self._prev_metrics, log_metrics = metrics, self._prev_metrics
        self._prev_step, log_step = step, self._prev_step
        if log_metrics is None:
            return
        assert log_step is not None
        print(f"{log_step}:", jax.tree.map(lambda x: x[-1].item(), log_metrics))
