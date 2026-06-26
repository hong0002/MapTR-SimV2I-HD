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
        pose_dim=7,
        hidden_dim=32,
        gate_bias=-1.5,
    ):
        super().__init__()
        self.ego_view_count = int(ego_view_count)
        self.rsu_view_count = int(rsu_view_count)
        self.pose_dim = int(pose_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, gate_bias)

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
        pose_feats = self._build_pose_features(
            img_metas,
            batch_size,
            rsu_count,
            device=ref_feat.device,
            dtype=ref_feat.dtype,
        )
        gates = torch.sigmoid(self.gate_mlp(pose_feats).squeeze(-1))

        gated_feats = []
        view_start = self.ego_view_count
        view_end = view_start + rsu_count
        for feat in img_feats:
            gated = feat.clone()
            gate_values = gates.to(dtype=feat.dtype)
            if feat.dim() == 5:
                gated[:, view_start:view_end] = (
                    gated[:, view_start:view_end]
                    * gate_values.view(batch_size, rsu_count, 1, 1, 1)
                )
            else:
                gated[:, :, view_start:view_end] = (
                    gated[:, :, view_start:view_end]
                    * gate_values.view(batch_size, 1, rsu_count, 1, 1, 1)
                )
            gated_feats.append(gated)

        stats = None
        if collect_stats:
            stats = self._summarize(
                gates,
                pose_feats,
                ref_feat,
                num_views,
                rsu_count,
            )
        return gated_feats, stats

    def _build_pose_features(self, img_metas, batch_size, rsu_count, device, dtype):
        pose_values = []
        for batch_idx in range(batch_size):
            meta = img_metas[batch_idx] if img_metas is not None else {}
            matrices = self._get_meta_list(meta, 'camera2ego')
            groups = self._get_meta_list(meta, 'selected_camera_groups')
            sample_values = []
            for rsu_idx in range(rsu_count):
                view_idx = self.ego_view_count + rsu_idx
                camera2ego = self._matrix_or_identity(matrices, view_idx)
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
            pose_values.append(sample_values)

        pose_array = np.asarray(pose_values, dtype=np.float32)
        return torch.as_tensor(pose_array, device=device, dtype=dtype)

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
            return np.eye(4, dtype=np.float32)
        matrix = np.asarray(matrices[index], dtype=np.float32)
        if matrix.shape != (4, 4):
            return np.eye(4, dtype=np.float32)
        return matrix

    def _summarize(self, gates, pose_feats, ref_feat, num_views, rsu_count):
        gates_detached = gates.detach()
        pose_detached = pose_feats.detach()
        return dict(
            gate_mean=float(gates_detached.mean().cpu()),
            gate_min=float(gates_detached.min().cpu()),
            gate_max=float(gates_detached.max().cpu()),
            gate_shape=tuple(gates.shape),
            pose_abs_mean=float(pose_detached.abs().mean().cpu()),
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
