"""Framework-agnostic early stopping for plain PyTorch loops (Lightning-style semantics)."""
from __future__ import annotations

import copy
import math
from typing import Any, Literal

import torch
import torch.distributed as dist


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        mode: "min" (loss-like) or "max" (accuracy-like).
        patience: Number of checks with no improvement before stopping.
        min_delta: Minimum change to count as an improvement (always a positive number).
        baseline: Metric must beat this value before the patience counter can start.
        stopping_threshold: Stop immediately once the metric is this good (or better).
        divergence_threshold: Stop immediately once the metric is this bad (or worse).
        check_finite: Stop immediately if the metric is NaN/inf.
        restore_best_weights: Keep a copy of the best weights; call `restore(model)` after training.
        start_after: Ignore the first N checks (warmup), so early noise cannot trigger a stop.
        sync_distributed: In DDP, all-reduce the stop decision so every rank exits together.
    """

    def __init__(
        self,
        mode: Literal["min", "max"] = "min",
        patience: int = 3,
        min_delta: float = 0.0,
        baseline: float | None = None,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None,
        check_finite: bool = True,
        restore_best_weights: bool = True,
        start_after: int = 0,
        sync_distributed: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if patience < 0 or start_after < 0:
            raise ValueError("patience and start_after must be >= 0")
        self.mode = mode
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.baseline = baseline
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.check_finite = check_finite
        self.restore_best_weights = restore_best_weights
        self.start_after = start_after
        self.sync_distributed = sync_distributed

        self.best: float = math.inf if mode == "min" else -math.inf
        self.wait: int = 0
        self.num_checks: int = 0
        self.best_step: int | None = None
        self.stopped_step: int | None = None
        self.stop_reason: str | None = None
        self.should_stop: bool = False
        self._best_state: dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers
    def _is_better(self, current: float, reference: float) -> bool:
        if self.mode == "min":
            return current < reference - self.min_delta
        return current > reference + self.min_delta

    def _beats(self, current: float, threshold: float) -> bool:
        """True if `current` is at least as good as `threshold` (no min_delta)."""
        return current <= threshold if self.mode == "min" else current >= threshold

    def _worse_than(self, current: float, threshold: float) -> bool:
        return current >= threshold if self.mode == "min" else current <= threshold

    # ---------------------------------------------------------------------- API
    def step(self, value: float | torch.Tensor, model: torch.nn.Module | None = None,
             step: int | None = None) -> bool:
        """Report the monitored metric. Returns True if training should stop.

        Call once per validation run, *after* validation. Pass `model` if
        restore_best_weights is enabled so the best snapshot can be taken.
        """
        current = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
        self.num_checks += 1
        step = step if step is not None else self.num_checks
        stop, reason = False, None

        if self.check_finite and not math.isfinite(current):
            stop, reason = True, f"metric is not finite ({current})"
        elif self.num_checks <= self.start_after:
            pass  # warmup: observe but neither track nor stop
        else:
            if self._is_better(current, self.best):
                self.best, self.wait, self.best_step = current, 0, step
                if self.restore_best_weights and model is not None:
                    self._best_state = copy.deepcopy(
                        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    )
            elif self.baseline is None or self._beats(self.best, self.baseline) or self.best_step is not None:
                # Only count patience once something has been tracked and the baseline is met.
                if self.baseline is None or self._beats(self.best, self.baseline):
                    self.wait += 1
                    if self.wait > self.patience:
                        stop, reason = True, f"no improvement > {self.min_delta} for {self.wait} checks"

            if self.stopping_threshold is not None and self._beats(current, self.stopping_threshold):
                stop, reason = True, f"stopping_threshold {self.stopping_threshold} reached"
            if self.divergence_threshold is not None and self._worse_than(current, self.divergence_threshold):
                stop, reason = True, f"divergence_threshold {self.divergence_threshold} crossed"

        stop = self._sync(stop)
        if stop and not self.should_stop:
            self.should_stop, self.stopped_step = True, step
            self.stop_reason = reason or "stopped by another rank"
        return self.should_stop

    def restore(self, model: torch.nn.Module) -> bool:
        """Load the best weights into `model`. Returns False if none were saved."""
        if self._best_state is None:
            return False
        model.load_state_dict(self._best_state)
        return True

    def _sync(self, stop: bool) -> bool:
        if not (self.sync_distributed and dist.is_available() and dist.is_initialized()):
            return stop
        device = torch.device("cuda", torch.cuda.current_device()) \
            if dist.get_backend() == "nccl" else torch.device("cpu")
        flag = torch.tensor(int(stop), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)  # any rank says stop -> all stop
        return bool(flag.item())

    # -------------------------------------------------------------- checkpointing
    def state_dict(self) -> dict[str, Any]:
        """Include this in your checkpoint so a resumed run keeps its patience counter."""
        return {
            "best": self.best, "wait": self.wait, "num_checks": self.num_checks,
            "best_step": self.best_step, "stopped_step": self.stopped_step,
            "stop_reason": self.stop_reason, "should_stop": self.should_stop,
            "best_state": self._best_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best = state["best"]
        self.wait = state["wait"]
        self.num_checks = state["num_checks"]
        self.best_step = state["best_step"]
        self.stopped_step = state["stopped_step"]
        self.stop_reason = state["stop_reason"]
        self.should_stop = state["should_stop"]
        self._best_state = state["best_state"]


# ------------------------------------------------------------------ usage example
if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    stopper = EarlyStopping(mode="min", patience=2, min_delta=1e-3)

    fake_val_losses = [1.0, 0.8, 0.7, 0.71, 0.705, 0.72, 0.69, 0.5]
    for epoch, val_loss in enumerate(fake_val_losses):
        # ... train_one_epoch(model); val_loss = validate(model) ...
        if stopper.step(val_loss, model=model, step=epoch):
            print(f"Stopped at epoch {epoch}: {stopper.stop_reason} (best={stopper.best:.3f} @ {stopper.best_step})")
            break
    stopper.restore(model)