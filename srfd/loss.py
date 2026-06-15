"""SR-FD EMA/queue loss for trainer-side use.

Behaviour summary:

* For each enabled extractor, holds either an EMA estimate or a feature queue
  estimate of the generated distribution's first and second moments.
* On every active training step, computes representation vectors from the
  current batch, blends them into the EMA buffers, computes the resulting
  Fréchet distance against the precomputed real-corpus statistics, and
  returns a normalized loss term per representation.
* The EMA buffers and feature queues are detached after every active step, so
  the autograd graph never grows across steps. In queue mode, queued features
  are detached but current-batch features remain in the moment computation and
  receive gradient.

Important:

* Pass ``beta`` close to ``1.0`` (e.g. ``0.999``) to track a smooth estimate
  of the generator distribution rather than the per-batch one.
* Set ``warmup_steps`` to skip SR-FD until the rest of the training loop has
  stabilized. During warmup the loss returns zero and the EMA is not updated.
* ``every_n_steps`` lets you cut SR-FD frequency to save compute; the loss is
  zero on inactive steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .extractors import BaseSRFDExtractor
from .frechet import frechet_distance
from .moments import (
    batch_mean_and_second_moment,
    covariance_from_mean_and_second_moment,
)


class SRFDEmaLoss(nn.Module):
    def __init__(
        self,
        extractors: List[BaseSRFDExtractor],
        real_stats: Any,
        negative_stats: Any = None,
        beta: float = 0.999,
        eps: float = 1e-6,
        normalize: bool = True,
        warmup_steps: int = 0,
        every_n_steps: int = 1,
        conditioner: Optional[Dict[str, Any]] = None,
        ema_per_timestep: bool = False,
        normalize_total_weight: bool = False,
        target_total_weight: float = 1.0,
        target_mode: str = "weighted",
        target_temperature: float = 1.0,
        stats_mode: str = "ema",
        queue_size: int = 50_000,
        queue_warmup_size: int = 0,
        queue_include_current: bool = True,
        queue_update: bool = True,
        active_until_step: Optional[int] = None,
        negative_weight: float = 0.0,
    ):
        super().__init__()
        if not extractors:
            raise ValueError("SRFDEmaLoss: at least one extractor is required.")

        def _parse_targets(raw_stats: Any, *, role: str) -> List[Dict[str, Any]]:
            raw_targets = raw_stats if isinstance(raw_stats, list) else [raw_stats]
            parsed: List[Dict[str, Any]] = []
            for idx, target_obj in enumerate(raw_targets):
                if target_obj is None:
                    continue
                if isinstance(target_obj, dict) and "stats" in target_obj:
                    stats = target_obj["stats"]
                    target_name = str(target_obj.get("name", f"{role}{idx}"))
                    target_weight = float(target_obj.get("weight", 1.0))
                else:
                    stats = target_obj
                    target_name = f"{role}{idx}"
                    target_weight = float(target_obj.get("target_weight", 1.0)) if isinstance(target_obj, dict) else 1.0
                if not isinstance(stats, dict) or "reps" not in stats:
                    raise ValueError("SRFDEmaLoss: every stats target must contain 'reps'.")
                safe_name = self._safe_key(f"{role}{idx}_{target_name}") or f"{role}{idx}"
                parsed.append(
                    {
                        "index": idx,
                        "name": target_name,
                        "safe_name": safe_name,
                        "weight": target_weight,
                        "stats": stats,
                        "reps": stats["reps"],
                    }
                )
            return parsed

        self._targets: List[Dict[str, Any]] = _parse_targets(real_stats, role="target")
        self._negative_targets: List[Dict[str, Any]] = (
            _parse_targets(negative_stats, role="negative") if negative_stats is not None else []
        )
        if not self._targets:
            raise ValueError("SRFDEmaLoss: at least one real_stats target is required.")
        self.negative_weight = max(float(negative_weight), 0.0)
        if self.negative_weight > 0.0 and not self._negative_targets:
            raise ValueError("SRFDEmaLoss: negative_weight > 0 requires negative_stats.")

        self.extractors = nn.ModuleList(extractors)
        self.beta = float(beta)
        self.eps = float(eps)
        self.normalize = bool(normalize)
        self.warmup_steps = int(warmup_steps)
        self.every_n_steps = max(int(every_n_steps), 1)
        self.ema_per_timestep = bool(ema_per_timestep)
        self.normalize_total_weight = bool(normalize_total_weight)
        self.target_total_weight = float(target_total_weight)
        self.target_mode = str(target_mode)
        if self.target_mode not in {"weighted", "min", "softmin"}:
            raise ValueError("SRFDEmaLoss: target_mode must be one of weighted, min, softmin.")
        self.target_temperature = max(float(target_temperature), self.eps)
        self.stats_mode = str(stats_mode).lower()
        if self.stats_mode not in {"ema", "queue"}:
            raise ValueError("SRFDEmaLoss: stats_mode must be one of 'ema' or 'queue'.")
        self.queue_size = max(int(queue_size), 1)
        self.queue_warmup_size = max(int(queue_warmup_size), 0)
        self.queue_include_current = bool(queue_include_current)
        self.queue_update = bool(queue_update)
        self.active_until_step = None if active_until_step is None else int(active_until_step)
        conditioner = conditioner or {}
        self.cond_enabled = bool(conditioner.get("enabled", False))
        self.cond_min_count = int(conditioner.get("min_count", 64))
        self.cond_shrinkage_tau = float(conditioner.get("shrinkage_tau", 128.0))
        self.cond_lambda = float(conditioner.get("lambda_cond", 1.0))
        self.cond_fallback = str(conditioner.get("fallback", "global"))
        self.cond_queue_size = max(
            int(conditioner.get("queue_size", min(self.queue_size, 4096))),
            1,
        )
        self.cond_queue_warmup_size = max(
            int(conditioner.get("queue_warmup_size", self.queue_warmup_size)),
            0,
        )
        self._dynamic_ema: Dict[str, Dict[str, Any]] = {}
        self._feature_queues: Dict[str, Dict[str, Any]] = {}
        self._raw_real_reps = self._targets[0]["reps"]

        self._real_mu: Dict[str, torch.Tensor] = {}
        self._real_cov: Dict[str, torch.Tensor] = {}
        self._target_buffer_names: Dict[Tuple[int, str], Tuple[str, str]] = {}
        self._negative_target_buffer_names: Dict[Tuple[int, str], Tuple[str, str]] = {}
        for ext in self.extractors:
            available_targets = [target for target in self._targets if ext.name in target["reps"]]
            if not available_targets:
                raise KeyError(
                    f"SRFDEmaLoss: reference stats missing entry for extractor '{ext.name}' in all targets."
                )

            primary_mu = None
            for target in available_targets:
                entry = target["reps"][ext.name]
                mu_r = entry["mu"].detach().to(torch.float32)
                cov_r = entry["cov"].detach().to(torch.float32)
                mu_name = f"real_mu__{target['safe_name']}__{ext.name}"
                cov_name = f"real_cov__{target['safe_name']}__{ext.name}"
                self.register_buffer(mu_name, mu_r, persistent=False)
                self.register_buffer(cov_name, cov_r, persistent=False)
                self._target_buffer_names[(int(target["index"]), ext.name)] = (mu_name, cov_name)

                if primary_mu is None:
                    primary_mu = mu_r
                    self.register_buffer(f"real_mu__{ext.name}", mu_r, persistent=False)
                    self.register_buffer(f"real_cov__{ext.name}", cov_r, persistent=False)
                    self._real_mu[ext.name] = mu_r
                    self._real_cov[ext.name] = cov_r

            for target in self._negative_targets:
                if ext.name not in target["reps"]:
                    continue
                entry = target["reps"][ext.name]
                mu_r = entry["mu"].detach().to(torch.float32)
                cov_r = entry["cov"].detach().to(torch.float32)
                mu_name = f"real_mu__{target['safe_name']}__{ext.name}"
                cov_name = f"real_cov__{target['safe_name']}__{ext.name}"
                self.register_buffer(mu_name, mu_r, persistent=False)
                self.register_buffer(cov_name, cov_r, persistent=False)
                self._negative_target_buffer_names[(int(target["index"]), ext.name)] = (mu_name, cov_name)

            # EMA buffers; lazily initialized on first active call.
            self.register_buffer(f"ema_mu__{ext.name}", torch.zeros_like(primary_mu), persistent=False)
            self.register_buffer(
                f"ema_M__{ext.name}",
                torch.zeros(primary_mu.size(0), primary_mu.size(0), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                f"ema_init__{ext.name}", torch.zeros(1, dtype=torch.bool), persistent=False
            )

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def _real(self, name: str) -> tuple:
        return (
            self.get_buffer(f"real_mu__{name}"),
            self.get_buffer(f"real_cov__{name}"),
        )

    def _target_infos(self, ext_name: str, *, negative: bool = False) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        targets = self._negative_targets if negative else self._targets
        buffer_names = self._negative_target_buffer_names if negative else self._target_buffer_names
        for target in targets:
            idx = int(target["index"])
            names = buffer_names.get((idx, ext_name))
            if names is None:
                continue
            mu_name, cov_name = names
            infos.append(
                {
                    "index": idx,
                    "name": target["name"],
                    "safe_name": target["safe_name"],
                    "weight": float(target["weight"]),
                    "raw_entry": target["reps"][ext_name],
                    "mu": self.get_buffer(mu_name),
                    "cov": self.get_buffer(cov_name),
                }
            )
        return infos

    def _ema(self, name: str) -> tuple:
        return (
            self.get_buffer(f"ema_mu__{name}"),
            self.get_buffer(f"ema_M__{name}"),
            self.get_buffer(f"ema_init__{name}"),
        )

    @staticmethod
    def _safe_key(key: str) -> str:
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in key)

    def _dynamic_state(
        self,
        ext_name: str,
        state_key: str,
        real_mu: torch.Tensor,
    ) -> Dict[str, Any]:
        key = f"{ext_name}::{state_key}"
        state = self._dynamic_ema.get(key)
        if state is None:
            state = {
                "mu": torch.zeros_like(real_mu, device=real_mu.device),
                "M": torch.zeros(
                    real_mu.size(0),
                    real_mu.size(0),
                    dtype=torch.float32,
                    device=real_mu.device,
                ),
                "init": False,
            }
            self._dynamic_ema[key] = state
        else:
            state["mu"] = state["mu"].to(real_mu.device)
            state["M"] = state["M"].to(real_mu.device)
        return state

    def _module_device(self) -> torch.device:
        for buffer in self.buffers():
            return buffer.device
        for parameter in self.parameters():
            return parameter.device
        return torch.device("cpu")

    def _queue_state(
        self,
        *,
        ext_name: str,
        state_key: Optional[str],
        feature_dim: int,
        device: torch.device,
        queue_size: int,
    ) -> Dict[str, Any]:
        key = f"{ext_name}::{state_key or 'global'}"
        size = max(int(queue_size), 1)
        state = self._feature_queues.get(key)
        if state is None:
            state = {
                "features": torch.zeros(size, feature_dim, dtype=torch.float32, device=device),
                "ptr": 0,
                "filled": 0,
                "size": size,
            }
            self._feature_queues[key] = state
            return state

        features = state["features"].to(device=device, dtype=torch.float32)
        filled = min(int(state.get("filled", 0)), int(features.size(0)))
        ptr = int(state.get("ptr", 0)) % max(int(features.size(0)), 1)
        if features.dim() != 2 or features.size(1) != feature_dim:
            features = torch.zeros(size, feature_dim, dtype=torch.float32, device=device)
            filled = 0
            ptr = 0
        elif features.size(0) != size:
            resized = torch.zeros(size, feature_dim, dtype=torch.float32, device=device)
            copy_n = min(filled, size)
            if copy_n > 0:
                resized[:copy_n].copy_(features[:copy_n])
            features = resized
            filled = copy_n
            ptr = filled % size

        state["features"] = features
        state["filled"] = filled
        state["ptr"] = ptr
        state["size"] = size
        return state

    def _append_queue(self, state: Dict[str, Any], rep: torch.Tensor) -> None:
        if not self.queue_update:
            return
        data = rep.detach().to(dtype=torch.float32)
        if data.dim() != 2 or data.size(0) == 0:
            return
        features = state["features"]
        size = int(state.get("size", features.size(0)))
        n = int(data.size(0))
        with torch.no_grad():
            if n >= size:
                features.copy_(data[-size:].to(features.device))
                state["ptr"] = 0
                state["filled"] = size
                return

            ptr = int(state.get("ptr", 0)) % size
            first = min(n, size - ptr)
            features[ptr : ptr + first].copy_(data[:first].to(features.device))
            rest = n - first
            if rest > 0:
                features[:rest].copy_(data[first:].to(features.device))
            state["ptr"] = (ptr + n) % size
            state["filled"] = min(size, int(state.get("filled", 0)) + n)

    def _update_queue_moments(
        self,
        *,
        ext: BaseSRFDExtractor,
        rep: torch.Tensor,
        state_key: Optional[str],
        queue_size: int,
        queue_warmup_size: Optional[int] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]:
        rep32 = rep.to(torch.float32)
        state = self._queue_state(
            ext_name=ext.name,
            state_key=state_key,
            feature_dim=int(rep32.size(1)),
            device=rep32.device,
            queue_size=queue_size,
        )
        filled_before = int(state.get("filled", 0))
        queued = state["features"][:filled_before].detach()
        current_count = int(rep32.size(0)) if self.queue_include_current else 0
        usable_count = filled_before + current_count
        warmup_size = self.queue_warmup_size if queue_warmup_size is None else max(int(queue_warmup_size), 0)
        if usable_count < warmup_size:
            self._append_queue(state, rep32)
            return None

        if self.queue_include_current:
            features = torch.cat([queued, rep32], dim=0) if filled_before > 0 else rep32
        else:
            if filled_before == 0:
                self._append_queue(state, rep32)
                return None
            features = queued

        mu, second_moment = batch_mean_and_second_moment(features)
        cov = covariance_from_mean_and_second_moment(mu, second_moment, eps=self.eps)
        self._append_queue(state, rep32)
        return mu, cov, filled_before, int(features.size(0))

    def _estimate_moments(
        self,
        *,
        ext: BaseSRFDExtractor,
        rep: torch.Tensor,
        state_key: Optional[str],
        static: bool,
        reference_mu: torch.Tensor,
        queue_size: Optional[int] = None,
        queue_warmup_size: Optional[int] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[int]]]:
        if self.stats_mode == "queue":
            queued = self._update_queue_moments(
                ext=ext,
                rep=rep,
                state_key=state_key if state_key is not None else "global",
                queue_size=queue_size or self.queue_size,
                queue_warmup_size=queue_warmup_size,
            )
            if queued is None:
                return None
            gen_mu, gen_cov, queue_fill, queue_used = queued
            return gen_mu, gen_cov, queue_fill, queue_used

        gen_mu, gen_cov = self._update_ema(
            ext=ext,
            rep=rep,
            state_key=state_key,
            static=static,
            reference_mu=reference_mu,
        )
        return gen_mu, gen_cov, None, None

    def auxiliary_state_dict(self) -> Dict[str, Any]:
        static_ema: Dict[str, Dict[str, Any]] = {}
        for ext in self.extractors:
            ema_mu, ema_M, ema_init = self._ema(ext.name)
            static_ema[ext.name] = {
                "mu": ema_mu.detach().cpu(),
                "M": ema_M.detach().cpu(),
                "init": bool(ema_init.item()),
            }
        dynamic_ema = {
            key: {
                "mu": value["mu"].detach().cpu(),
                "M": value["M"].detach().cpu(),
                "init": bool(value.get("init", False)),
            }
            for key, value in self._dynamic_ema.items()
        }
        feature_queues = {
            key: {
                "features": value["features"].detach().cpu(),
                "ptr": int(value.get("ptr", 0)),
                "filled": int(value.get("filled", 0)),
                "size": int(value.get("size", value["features"].size(0))),
            }
            for key, value in self._feature_queues.items()
        }
        return {
            "format_version": 1,
            "stats_mode": self.stats_mode,
            "static_ema": static_ema,
            "dynamic_ema": dynamic_ema,
            "feature_queues": feature_queues,
        }

    def load_auxiliary_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        device = self._module_device()
        for ext_name, value in state.get("static_ema", {}).items():
            if f"ema_mu__{ext_name}" not in self._buffers or f"ema_M__{ext_name}" not in self._buffers:
                continue
            ema_mu, ema_M, ema_init = self._ema(ext_name)
            if tuple(ema_mu.shape) == tuple(value["mu"].shape):
                ema_mu.copy_(value["mu"].to(ema_mu.device, dtype=torch.float32))
            if tuple(ema_M.shape) == tuple(value["M"].shape):
                ema_M.copy_(value["M"].to(ema_M.device, dtype=torch.float32))
            ema_init.fill_(bool(value.get("init", False)))

        self._dynamic_ema = {}
        for key, value in state.get("dynamic_ema", {}).items():
            self._dynamic_ema[str(key)] = {
                "mu": value["mu"].to(device=device, dtype=torch.float32),
                "M": value["M"].to(device=device, dtype=torch.float32),
                "init": bool(value.get("init", False)),
            }

        self._feature_queues = {}
        for key, value in state.get("feature_queues", {}).items():
            features = value["features"].to(device=device, dtype=torch.float32)
            size = int(value.get("size", features.size(0)))
            filled = min(int(value.get("filled", 0)), int(features.size(0)), size)
            self._feature_queues[str(key)] = {
                "features": features,
                "ptr": int(value.get("ptr", 0)) % max(size, 1),
                "filled": filled,
                "size": size,
            }

    def _real_cond(self, ext_name: str, cond_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        safe = self._safe_key(f"{ext_name}__cond__{cond_key}")
        mu_name = f"real_mu__{safe}"
        cov_name = f"real_cov__{safe}"
        if mu_name not in self._buffers or cov_name not in self._buffers:
            return None
        # The sample count remains in the raw stats dict; keep this path cheap.
        return self.get_buffer(mu_name), self.get_buffer(cov_name), 0

    def _compute_fd_term(
        self,
        *,
        ext: BaseSRFDExtractor,
        rep: torch.Tensor,
        real_mu: torch.Tensor,
        real_cov: torch.Tensor,
        state_key: Optional[str],
        static: bool,
        extra_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_mu, cov_g = self._update_ema(
            ext=ext,
            rep=rep,
            state_key=state_key,
            static=static,
            reference_mu=real_mu,
        )
        return self._compute_fd_from_moments(
            ext=ext,
            real_mu=real_mu,
            real_cov=real_cov,
            gen_mu=new_mu,
            gen_cov=cov_g,
            extra_weight=extra_weight,
        )

    def _update_ema(
        self,
        *,
        ext: BaseSRFDExtractor,
        rep: torch.Tensor,
        state_key: Optional[str],
        static: bool,
        reference_mu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_batch, M_batch = batch_mean_and_second_moment(rep)

        if static:
            ema_mu, ema_M, ema_init = self._ema(ext.name)
            initialized = bool(ema_init.item())
        else:
            state = self._dynamic_state(ext.name, state_key or "global", reference_mu)
            ema_mu, ema_M = state["mu"], state["M"]
            initialized = bool(state["init"])

        beta = self.beta
        if not initialized:
            new_mu = mu_batch
            new_M = M_batch
        else:
            new_mu = beta * ema_mu.detach() + (1.0 - beta) * mu_batch
            new_M = beta * ema_M.detach() + (1.0 - beta) * M_batch

        cov_g = covariance_from_mean_and_second_moment(new_mu, new_M, eps=self.eps)
        if static:
            ema_mu.copy_(new_mu.detach())
            ema_M.copy_(new_M.detach())
            ema_init.fill_(True)
        else:
            state["mu"] = new_mu.detach()
            state["M"] = new_M.detach()
            state["init"] = True
        return new_mu, cov_g

    def _compute_fd_from_moments(
        self,
        *,
        ext: BaseSRFDExtractor,
        real_mu: torch.Tensor,
        real_cov: torch.Tensor,
        gen_mu: torch.Tensor,
        gen_cov: torch.Tensor,
        extra_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fd_k = frechet_distance(real_mu, real_cov, gen_mu, gen_cov, eps=self.eps)
        if self.normalize:
            loss_k = ext.weight * extra_weight * fd_k / (fd_k.detach() + self.eps)
        else:
            loss_k = ext.weight * extra_weight * fd_k
        return loss_k, fd_k

    def _combine_target_terms(
        self,
        *,
        ext: BaseSRFDExtractor,
        gen_mu: torch.Tensor,
        gen_cov: torch.Tensor,
        targets: List[Dict[str, Any]],
        extra_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        losses = []
        fds = []
        raw_weights = []
        target_logs: Dict[str, torch.Tensor] = {}
        for info in targets:
            loss_t, fd_t = self._compute_fd_from_moments(
                ext=ext,
                real_mu=info["mu"],
                real_cov=info["cov"],
                gen_mu=gen_mu,
                gen_cov=gen_cov,
                extra_weight=extra_weight,
            )
            losses.append(loss_t)
            fds.append(fd_t)
            raw_weights.append(max(float(info["weight"]), 0.0))
            target_logs[f"target_{info['safe_name']}"] = fd_t.detach()

        if not losses:
            raise RuntimeError("SRFDEmaLoss: _combine_target_terms called without targets.")
        loss_stack = torch.stack(losses)
        fd_stack = torch.stack(fds)
        weight_tensor = torch.as_tensor(raw_weights, dtype=torch.float32, device=fd_stack.device)
        if weight_tensor.sum().item() <= 0.0:
            weight_tensor = torch.ones_like(weight_tensor)

        if self.target_mode == "min":
            idx = int(torch.argmin(fd_stack.detach()).item())
            weights = torch.zeros_like(weight_tensor)
            weights[idx] = 1.0
        elif self.target_mode == "softmin":
            logits = -fd_stack.detach() / self.target_temperature
            weights = torch.softmax(logits, dim=0) * weight_tensor
            weights = weights / weights.sum().clamp_min(self.eps)
        else:
            weights = weight_tensor / weight_tensor.sum().clamp_min(self.eps)

        combined_loss = (loss_stack * weights).sum()
        combined_fd = (fd_stack.detach() * weights).sum()
        return combined_loss, combined_fd, target_logs

    def _conditioned_real_stats(
        self,
        raw_entry: Dict[str, Any],
        cond_key: str,
        real_mu: torch.Tensor,
        real_cov: torch.Tensor,
        *,
        min_count: Optional[int] = None,
        shrinkage_tau: Optional[float] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        cond = raw_entry.get("cond", {}) if isinstance(raw_entry, dict) else {}
        cond_entry = cond.get(cond_key)
        if not cond_entry:
            return None
        n = int(cond_entry.get("n", 0))
        min_count = self.cond_min_count if min_count is None else int(min_count)
        shrinkage_tau = self.cond_shrinkage_tau if shrinkage_tau is None else float(shrinkage_tau)
        if n < min_count:
            return None
        alpha = n / (n + shrinkage_tau)
        mu_c = cond_entry["mu"].to(real_mu.device, dtype=torch.float32)
        cov_c = cond_entry["cov"].to(real_cov.device, dtype=torch.float32)
        mu = alpha * mu_c + (1.0 - alpha) * real_mu
        cov = alpha * cov_c + (1.0 - alpha) * real_cov
        return mu, cov, n

    def _condition_groups_from_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.cond_enabled:
            return []
        raw_groups = batch.get("condition_key_groups", None)
        if raw_groups:
            groups = []
            for i, group in enumerate(raw_groups):
                keys = list(group.get("keys", []))
                if not keys:
                    continue
                groups.append(
                    {
                        "name": self._safe_key(str(group.get("name", f"group{i}"))) or f"group{i}",
                        "keys": keys,
                        "lambda": float(group.get("lambda", self.cond_lambda)),
                        "min_count": int(group.get("min_count", self.cond_min_count)),
                        "shrinkage_tau": float(group.get("shrinkage_tau", self.cond_shrinkage_tau)),
                        "queue_warmup_size": int(group.get("queue_warmup_size", self.cond_queue_warmup_size)),
                    }
                )
            return groups

        if batch.get("condition_keys"):
            return [
                {
                    "name": "cond",
                    "keys": list(batch["condition_keys"]),
                    "lambda": self.cond_lambda,
                    "min_count": self.cond_min_count,
                    "shrinkage_tau": self.cond_shrinkage_tau,
                    "queue_warmup_size": self.cond_queue_warmup_size,
                }
            ]
        return []

    @staticmethod
    def _filter_condition_groups(
        groups: List[Dict[str, Any]],
        active_index: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        if not groups:
            return []
        index_list = active_index.detach().to(device="cpu", dtype=torch.long).tolist()
        filtered: List[Dict[str, Any]] = []
        for group in groups:
            keys = list(group.get("keys", []))
            if not keys:
                continue
            filtered_keys = [keys[i] for i in index_list if 0 <= int(i) < len(keys)]
            if not filtered_keys:
                continue
            group_copy = dict(group)
            group_copy["keys"] = filtered_keys
            filtered.append(group_copy)
        return filtered

    # ------------------------------------------------------------------ #
    # Active step gating
    # ------------------------------------------------------------------ #
    def is_active(self, step: int) -> bool:
        if step < self.warmup_steps:
            return False
        if self.active_until_step is not None and step > self.active_until_step:
            return False
        if (step - self.warmup_steps) % self.every_n_steps != 0:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        step: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        # Pick a device/dtype anchor from the input so zero losses come out on
        # the right device when SR-FD is inactive.
        anchor = None
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                anchor = v
                break
        if device is None and anchor is not None:
            device = anchor.device
        zero = torch.zeros((), dtype=torch.float32, device=device or "cpu")

        if not self.is_active(step):
            return {"loss/srfd": zero}

        outputs: Dict[str, torch.Tensor] = {}
        sample_weight_raw = batch.get("sample_weight", None)
        if sample_weight_raw is not None and not isinstance(sample_weight_raw, torch.Tensor):
            sample_weight_raw = torch.as_tensor(sample_weight_raw, dtype=torch.float32, device=zero.device)
        sample_weight_by_extractor = batch.get("sample_weight_by_extractor", {}) or {}
        if sample_weight_by_extractor and not isinstance(sample_weight_by_extractor, dict):
            raise ValueError("SRFDEmaLoss sample_weight_by_extractor must be a dict when provided.")
        total_loss = zero.clone()
        total_weight = zero.clone()
        any_term = False

        for ext in self.extractors:
            try:
                rep = ext(batch).to(torch.float32)
            except KeyError:
                # Extractor's required keys are missing; skip silently so a
                # malformed batch doesn't kill training.
                continue
            if rep.dim() != 2 or rep.size(0) == 0:
                continue

            active_index = None
            sample_loss_scale = rep.new_tensor(1.0, dtype=torch.float32)
            ext_sample_weight_raw = sample_weight_raw
            if sample_weight_by_extractor:
                safe_ext_name = self._safe_key(ext.name)
                ext_sample_weight_raw = sample_weight_by_extractor.get(
                    ext.name,
                    sample_weight_by_extractor.get(safe_ext_name, sample_weight_raw),
                )
            if ext_sample_weight_raw is not None:
                if not isinstance(ext_sample_weight_raw, torch.Tensor):
                    ext_sample_weight_raw = torch.as_tensor(
                        ext_sample_weight_raw,
                        dtype=torch.float32,
                        device=rep.device,
                    )
                sample_weight = ext_sample_weight_raw.to(device=rep.device, dtype=torch.float32).flatten()
                if sample_weight.numel() != rep.size(0):
                    raise ValueError(
                        "SRFDEmaLoss sample_weight batch mismatch: "
                        f"{sample_weight.numel()} vs {rep.size(0)} for extractor {ext.name}"
                    )
                active = sample_weight > self.eps
                outputs[f"srfd/sample_weight_mean_{ext.name}"] = sample_weight.mean().detach()
                outputs[f"srfd/sample_weight_sum_{ext.name}"] = sample_weight.sum().detach()
                outputs[f"srfd/sample_weight_active_{ext.name}"] = active.to(torch.float32).mean().detach()
                if not bool(active.any().item()):
                    continue
                active_index = active.nonzero(as_tuple=False).flatten()
                sample_loss_scale = sample_weight.index_select(0, active_index).mean().clamp_min(self.eps)
                if active_index.numel() != rep.size(0):
                    rep = rep.index_select(0, active_index)

            target_infos = self._target_infos(ext.name)
            if not target_infos:
                continue
            nfe = batch.get("nfe", None)
            nfe_int = int(nfe) if nfe is not None else None
            global_state_key = f"global::nfe{nfe_int}" if self.ema_per_timestep and nfe_int is not None else None
            moments = self._estimate_moments(
                ext=ext,
                rep=rep,
                state_key=global_state_key,
                static=global_state_key is None,
                reference_mu=target_infos[0]["mu"],
                queue_size=self.queue_size,
                queue_warmup_size=self.queue_warmup_size,
            )
            if moments is None:
                continue
            gen_mu, gen_cov, queue_fill, queue_used = moments
            loss_k, fd_k, target_logs = self._combine_target_terms(
                ext=ext,
                gen_mu=gen_mu,
                gen_cov=gen_cov,
                targets=target_infos,
                extra_weight=sample_loss_scale,
            )
            neg_loss_k = None
            neg_fd_k = None
            neg_target_logs: Dict[str, torch.Tensor] = {}
            if self.negative_weight > 0.0:
                negative_infos = self._target_infos(ext.name, negative=True)
                if negative_infos:
                    neg_loss_k, neg_fd_k, neg_target_logs = self._combine_target_terms(
                        ext=ext,
                        gen_mu=gen_mu,
                        gen_cov=gen_cov,
                        targets=negative_infos,
                        extra_weight=sample_loss_scale,
                    )
                    loss_k = loss_k - float(self.negative_weight) * neg_loss_k

            total_loss = total_loss + loss_k
            total_weight = total_weight + float(ext.weight) * (1.0 + float(self.negative_weight))
            any_term = True

            outputs[f"srfd/fd_{ext.name}"] = fd_k.detach()
            outputs[f"srfd/loss_{ext.name}"] = loss_k.detach()
            if neg_loss_k is not None and neg_fd_k is not None:
                outputs[f"srfd/fd_{ext.name}_negative"] = neg_fd_k.detach()
                outputs[f"srfd/loss_{ext.name}_negative_repel"] = neg_loss_k.detach()
            if queue_fill is not None:
                outputs[f"srfd/queue_fill_{ext.name}"] = torch.as_tensor(
                    queue_fill, dtype=torch.float32, device=loss_k.device
                )
                outputs[f"srfd/queue_used_{ext.name}"] = torch.as_tensor(
                    queue_used or 0, dtype=torch.float32, device=loss_k.device
                )
            for target_key, target_fd in target_logs.items():
                outputs[f"srfd/fd_{ext.name}_{target_key}"] = target_fd
            for target_key, target_fd in neg_target_logs.items():
                outputs[f"srfd/fd_{ext.name}_negative_{target_key}"] = target_fd

            condition_groups = self._condition_groups_from_batch(batch)
            if active_index is not None:
                condition_groups = self._filter_condition_groups(condition_groups, active_index)
            if condition_groups:
                cond_losses = []
                cond_fds = []
                cond_weight = 0.0
                for group in condition_groups:
                    group_keys = list(group["keys"])
                    group_lambda = float(group.get("lambda", self.cond_lambda))
                    group_name = str(group.get("name", "cond"))
                    for cond_key in sorted(set(group_keys)):
                        idx = [i for i, key in enumerate(group_keys) if key == cond_key]
                        if not idx:
                            continue
                        cond_targets = []
                        for target_info in target_infos:
                            cond_real = self._conditioned_real_stats(
                                target_info["raw_entry"],
                                cond_key,
                                target_info["mu"],
                                target_info["cov"],
                                min_count=int(group.get("min_count", self.cond_min_count)),
                                shrinkage_tau=float(group.get("shrinkage_tau", self.cond_shrinkage_tau)),
                            )
                            if cond_real is None:
                                continue
                            mu_c, cov_c, _n_c = cond_real
                            cond_target = dict(target_info)
                            cond_target["mu"] = mu_c
                            cond_target["cov"] = cov_c
                            cond_targets.append(cond_target)
                        if not cond_targets:
                            continue
                        index = torch.tensor(idx, device=rep.device, dtype=torch.long)
                        rep_c = rep.index_select(0, index)
                        cond_state_key = (
                            f"cond_group={group_name}::cond={cond_key}::nfe{nfe_int}"
                            if nfe_int is not None
                            else f"cond_group={group_name}::cond={cond_key}"
                        )
                        cond_moments = self._estimate_moments(
                            ext=ext,
                            rep=rep_c,
                            state_key=cond_state_key,
                            static=False,
                            reference_mu=cond_targets[0]["mu"],
                            queue_size=self.cond_queue_size,
                            queue_warmup_size=int(group.get("queue_warmup_size", self.cond_queue_warmup_size)),
                        )
                        if cond_moments is None:
                            continue
                        cond_gen_mu, cond_gen_cov, _cond_queue_fill, _cond_queue_used = cond_moments
                        cond_loss, cond_fd, _cond_target_logs = self._combine_target_terms(
                            ext=ext,
                            gen_mu=cond_gen_mu,
                            gen_cov=cond_gen_cov,
                            targets=cond_targets,
                            extra_weight=group_lambda,
                        )
                        cond_losses.append(cond_loss)
                        cond_fds.append(cond_fd.detach())
                        cond_weight += float(ext.weight) * group_lambda
                if cond_losses:
                    cond_loss_total = torch.stack(cond_losses).sum()
                    total_loss = total_loss + cond_loss_total
                    total_weight = total_weight + cond_weight
                    outputs[f"srfd/loss_{ext.name}_cond"] = cond_loss_total.detach()
                    outputs[f"srfd/fd_{ext.name}_cond_mean"] = torch.stack(cond_fds).mean()

        if not any_term:
            return {"loss/srfd": zero}

        if self.normalize_total_weight:
            normalizer = total_weight.detach().clamp_min(self.eps)
            total_loss = total_loss / normalizer * self.target_total_weight
            outputs["srfd/active_weight_sum"] = total_weight.detach()
            outputs["srfd/target_total_weight"] = torch.as_tensor(
                self.target_total_weight, dtype=torch.float32, device=total_loss.device
            )

        outputs["loss/srfd"] = total_loss
        return outputs
