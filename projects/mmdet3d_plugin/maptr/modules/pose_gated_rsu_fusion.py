import math

import numpy as np
import torch
import torch.nn as nn


class PoseAwareGatedRSUFusion(nn.Module):
    """Pose-conditioned feature gate for dynamic RSU camera views.

    This module is intentionally lightweight: it keeps the ego camera features
    unchanged and scales only the RSU camera features before MapTR's existing
    transformer consumes the multi-view feature list.
    """

    def __init__(
        self,
        ego_view_count=6,
        rsu_view_count=4,
        gate_mode='pose',
        use_pose_metadata=None,
        constant_gate_value=0.2,
        nopose_gate_init=0.0,
        pose_dim=7,
        hidden_dim=32,
        gate_bias=-1.5,
        fallback_gate=0.2,
    ):
        super().__init__()
        self.ego_view_count = int(ego_view_count)
        self.rsu_view_count = int(rsu_view_count)
        self.gate_mode = str(gate_mode).lower()
        if self.gate_mode not in ('pose', 'constant', 'nopose'):
            raise ValueError(
                "Unsupported gate_mode '{}'. Expected one of: pose, "
                "constant, nopose.".format(gate_mode)
            )
        if use_pose_metadata is None:
            use_pose_metadata = self.gate_mode == 'pose'
        self.use_pose_metadata = bool(use_pose_metadata)
        if self.gate_mode == 'pose' and not self.use_pose_metadata:
            raise ValueError("gate_mode='pose' requires use_pose_metadata=True.")
        if self.gate_mode != 'pose' and self.use_pose_metadata:
            raise ValueError(
                "gate_mode='{}' must not use pose metadata.".format(self.gate_mode)
            )
        self.constant_gate_value = float(constant_gate_value)
        self.pose_dim = int(pose_dim)
        self.fallback_gate = float(fallback_gate)

        self.gate_mlp = None
        self.rsu_slot_gate = None
        if self.gate_mode == 'pose':
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.pose_dim, hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.constant_(self.gate_mlp[-1].bias, gate_bias)
        elif self.gate_mode == 'nopose':
            self.rsu_slot_gate = nn.Parameter(
                torch.full((self.rsu_view_count,), float(nopose_gate_init))
            )

    def forward(self, img_feats, img_metas, collect_stats=False):
        if img_feats is None or len(img_feats) == 0:
            return img_feats, None

        ref_feat = img_feats[0]
        if ref_feat.dim() not in (5, 6):
            raise ValueError(
                'Expected feature shape [B,N,C,H,W] or [B,L,N,C,H,W], got '
                '{}.'.format(tuple(ref_feat.shape))
            )

        num_views = ref_feat.shape[1] if ref_feat.dim() == 5 else ref_feat.shape[2]
        rsu_count = min(self.rsu_view_count, max(0, num_views - self.ego_view_count))
        if rsu_count <= 0:
            return img_feats, None

        batch_size = ref_feat.shape[0]
        pose_feats = None
        fallback_mask = torch.zeros(
            (batch_size, rsu_count),
            device=ref_feat.device,
            dtype=torch.bool,
        )
        if self.gate_mode == 'pose':
            mlp_dtype = next(self.gate_mlp.parameters()).dtype
            pose_feats, fallback_mask = self._build_pose_features(
                img_metas,
                batch_size,
                rsu_count,
                device=ref_feat.device,
                dtype=mlp_dtype,
            )
            gates = torch.sigmoid(self.gate_mlp(pose_feats).squeeze(-1))
            fallback_values = gates.new_full(gates.shape, self.fallback_gate)
            gates = torch.where(fallback_mask, fallback_values, gates)
        elif self.gate_mode == 'constant':
            gates = ref_feat.new_full(
                (batch_size, rsu_count),
                self.constant_gate_value,
            )
            fallback_values = gates.new_full(gates.shape, self.constant_gate_value)
        else:
            gate_logits = self.rsu_slot_gate[:rsu_count].to(device=ref_feat.device)
            gates = torch.sigmoid(gate_logits).unsqueeze(0).expand(batch_size, -1)
            fallback_values = gates.new_full(gates.shape, self.fallback_gate)
        gates = torch.clamp(gates, min=0.0, max=1.0)
        gates = torch.where(torch.isfinite(gates), gates, fallback_values)

        gated_feats = []
        view_start = self.ego_view_count
        view_end = view_start + rsu_count
        for feat in img_feats:
            gate_values = gates.to(device=feat.device, dtype=feat.dtype)
            if feat.dim() == 5:
                ego_feat = feat[:, :view_start]
                rsu_feat = feat[:, view_start:view_end]
                tail_feat = feat[:, view_end:]
                gate_view = gate_values.view(batch_size, rsu_count, 1, 1, 1)
                rsu_feat = rsu_feat * gate_view
                gated = torch.cat([ego_feat, rsu_feat, tail_feat], dim=1)
            else:
                ego_feat = feat[:, :, :view_start]
                rsu_feat = feat[:, :, view_start:view_end]
                tail_feat = feat[:, :, view_end:]
                gate_view = gate_values.view(batch_size, 1, rsu_count, 1, 1, 1)
                rsu_feat = rsu_feat * gate_view
                gated = torch.cat([ego_feat, rsu_feat, tail_feat], dim=2)
            gated_feats.append(gated)

        stats = None
        if collect_stats:
            stats = self._summarize(
                gates,
                pose_feats,
                fallback_mask,
                ref_feat,
                num_views,
                rsu_count,
            )
        return gated_feats, stats

    def _build_pose_features(self, img_metas, batch_size, rsu_count, device, dtype):
        pose_values = []
        fallback_values = []
        for batch_idx in range(batch_size):
            meta = self._normalize_meta(img_metas, batch_idx)
            matrices = self._get_meta_list(meta, 'camera2ego')
            if not matrices:
                matrices = self._get_meta_list(meta, 'cam2ego')
            if not matrices:
                matrices = self._get_meta_list(meta, 'sensor2ego')
            if not matrices:
                matrices = self._get_meta_list(meta, 'lidar2img')
            groups = self._get_meta_list(meta, 'selected_camera_groups')
            sample_values = []
            sample_fallbacks = []
            for rsu_idx in range(rsu_count):
                view_idx = self.ego_view_count + rsu_idx
                camera2ego, used_fallback = self._matrix_or_identity(
                    matrices, view_idx
                )
                translation = camera2ego[:3, 3]
                rotation = camera2ego[:3, :3]
                yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
                distance = math.sqrt(
                    float(translation[0]) ** 2 + float(translation[1]) ** 2
                )
                group_flag = 0.0
                if view_idx < len(groups) and str(groups[view_idx]).lower() == 'rsu':
                    group_flag = 1.0
                sample_values.append(
                    [
                        float(translation[0]) / 50.0,
                        float(translation[1]) / 50.0,
                        float(translation[2]) / 10.0,
                        math.sin(yaw),
                        math.cos(yaw),
                        distance / 75.0,
                        group_flag,
                    ]
                )
                sample_fallbacks.append(used_fallback)
            pose_values.append(sample_values)
            fallback_values.append(sample_fallbacks)

        pose_array = np.asarray(pose_values, dtype=np.float32)
        fallback_array = np.asarray(fallback_values, dtype=np.bool_)
        return (
            torch.as_tensor(pose_array, device=device, dtype=dtype),
            torch.as_tensor(fallback_array, device=device, dtype=torch.bool),
        )

    @staticmethod
    def _normalize_meta(img_metas, batch_idx):
        if img_metas is None:
            return {}
        if isinstance(img_metas, dict):
            return img_metas
        if not isinstance(img_metas, (list, tuple)) or len(img_metas) == 0:
            return {}
        item = img_metas[min(batch_idx, len(img_metas) - 1)]
        if isinstance(item, (list, tuple)) and len(item) > 0:
            item = item[-1]
        return item if isinstance(item, dict) else {}

    @staticmethod
    def _get_meta_list(meta, key):
        if not isinstance(meta, dict) or key not in meta:
            return []
        value = meta[key]
        if isinstance(value, tuple):
            return list(value)
        return value

    @staticmethod
    def _matrix_or_identity(matrices, index):
        if matrices is None or index >= len(matrices):
            return np.eye(4, dtype=np.float32), True
        matrix = np.asarray(matrices[index], dtype=np.float32)
        if matrix.shape != (4, 4):
            return np.eye(4, dtype=np.float32), True
        return matrix, False

    def _summarize(
        self,
        gates,
        pose_feats,
        fallback_mask,
        ref_feat,
        num_views,
        rsu_count,
    ):
        gates_detached = gates.detach()
        fallback_detached = fallback_mask.detach()
        pose_abs_mean = None
        if pose_feats is not None:
            pose_abs_mean = float(pose_feats.detach().abs().mean().cpu())
        return dict(
            gate_mode=self.gate_mode,
            use_pose_metadata=self.use_pose_metadata,
            learnable_gate=self.gate_mode in ('pose', 'nopose'),
            constant_gate_value=self.constant_gate_value
            if self.gate_mode == 'constant'
            else None,
            gate_mean=float(gates_detached.mean().cpu()),
            gate_min=float(gates_detached.min().cpu()),
            gate_max=float(gates_detached.max().cpu()),
            gate_finite=bool(torch.isfinite(gates_detached).all().cpu()),
            gate_shape=tuple(gates.shape),
            fallback_count=int(fallback_detached.sum().cpu()),
            fallback_gate=self.fallback_gate,
            pose_abs_mean=pose_abs_mean,
            level0_shape=tuple(ref_feat.shape),
            ego_level0_shape=(
                ref_feat.shape[0],
                self.ego_view_count,
                *tuple(ref_feat.shape[-3:]),
            ),
            rsu_level0_shape=(
                ref_feat.shape[0],
                rsu_count,
                *tuple(ref_feat.shape[-3:]),
            ),
            num_views=int(num_views),
            rsu_count=int(rsu_count),
        )
