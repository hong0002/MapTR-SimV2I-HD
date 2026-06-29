import copy

import torch
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS

from projects.mmdet3d_plugin.maptr.modules.pose_gated_rsu_fusion import (
    PoseAwareGatedRSUFusion,
)

from .maptr import MapTR


@DETECTORS.register_module()
class MapTRPoseGatedV2I(MapTR):
    """MapTR with a pose-conditioned RSU feature gate.

    This is a feature-level MVP: ego camera features are preserved, RSU camera
    image features are pose-gated, and the existing MapTR transformer/decoder
    remains unchanged.
    """

    def __init__(
        self,
        pose_gate=None,
        ego_view_count=6,
        rsu_view_count=4,
        input_rsu_view_count=4,
        debug_gate=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        pose_gate = dict(pose_gate or {})
        pose_gate.pop('type', None)
        pose_gate.setdefault('ego_view_count', ego_view_count)
        pose_gate.setdefault('rsu_view_count', rsu_view_count)
        self.pose_gate = PoseAwareGatedRSUFusion(**pose_gate)
        self.ego_view_count = int(ego_view_count)
        self.rsu_view_count = int(rsu_view_count)
        self.input_rsu_view_count = int(input_rsu_view_count)
        self.debug_gate = bool(debug_gate)
        self._pose_gate_view_debug_printed = False
        self._pose_gate_debug_printed = False
        self._pose_gate_loss_debug_printed = False
        self.latest_pose_gate_stats = None

    @staticmethod
    def _is_rank0():
        if not torch.distributed.is_available():
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _view_indices(self, num_views):
        ego_count = min(self.ego_view_count, num_views)
        rsu_available = max(0, num_views - ego_count)
        rsu_count = min(self.rsu_view_count, rsu_available)
        return list(range(ego_count)) + list(
            range(ego_count, ego_count + rsu_count)
        )

    @staticmethod
    def _slice_meta_value(value, indices, original_view_count):
        if isinstance(value, list) and len(value) == original_view_count:
            return [value[index] for index in indices]
        if isinstance(value, tuple) and len(value) == original_view_count:
            return tuple(value[index] for index in indices)
        if hasattr(value, 'shape') and len(getattr(value, 'shape', ())) > 0:
            if value.shape[0] == original_view_count:
                return value[indices]
        return value

    def _slice_single_meta(self, meta, indices, original_view_count):
        if not isinstance(meta, dict):
            return meta
        sliced = {}
        for key, value in meta.items():
            sliced[key] = self._slice_meta_value(
                value,
                indices,
                original_view_count,
            )
        return sliced

    def _slice_img_metas(self, img_metas, indices, original_view_count):
        if img_metas is None:
            return img_metas
        if isinstance(img_metas, dict):
            return self._slice_single_meta(img_metas, indices, original_view_count)
        if not isinstance(img_metas, (list, tuple)):
            return img_metas
        sliced = [
            self._slice_single_meta(meta, indices, original_view_count)
            for meta in img_metas
        ]
        return tuple(sliced) if isinstance(img_metas, tuple) else sliced

    def _select_topk_views(self, img, img_metas=None, phase='train'):
        if img is None or img.dim() not in (5, 6):
            return img, img_metas

        view_dim = 1 if img.dim() == 5 else 2
        original_view_count = int(img.size(view_dim))
        indices = self._view_indices(original_view_count)
        used_view_count = len(indices)
        if used_view_count == original_view_count:
            selected_img = img
        else:
            selected_img = img.index_select(
                view_dim,
                torch.as_tensor(indices, device=img.device, dtype=torch.long),
            )
        selected_metas = self._slice_img_metas(
            img_metas,
            indices,
            original_view_count,
        )

        if (
            self.debug_gate
            and self._is_rank0()
            and not self._pose_gate_view_debug_printed
        ):
            metadata_view_count = None
            if isinstance(selected_metas, (list, tuple)) and selected_metas:
                meta0 = selected_metas[0]
                if isinstance(meta0, dict):
                    for key in ('camera2ego', 'lidar2img', 'filename'):
                        value = meta0.get(key)
                        if isinstance(value, (list, tuple)):
                            metadata_view_count = len(value)
                            break
            print(
                '[MapTRPoseGatedV2I] phase={} original_view_count={} '
                'used_view_count={} ego_view_count={} rsu_view_count={} '
                'metadata_view_count={} indices={}'.format(
                    phase,
                    original_view_count,
                    used_view_count,
                    min(self.ego_view_count, original_view_count),
                    max(0, used_view_count - min(self.ego_view_count, used_view_count)),
                    metadata_view_count,
                    indices,
                )
            )
            self._pose_gate_view_debug_printed = True
        return selected_img, selected_metas

    def _apply_pose_gate(self, img_feats, img_metas, phase):
        collect_stats = (
            self.debug_gate
            and self._is_rank0()
            and not self._pose_gate_debug_printed
        )
        img_feats, stats = self.pose_gate(
            img_feats,
            img_metas,
            collect_stats=collect_stats,
        )
        if stats is not None:
            self.latest_pose_gate_stats = stats
            meta_keys = []
            if isinstance(img_metas, (list, tuple)) and len(img_metas) > 0:
                if isinstance(img_metas[0], dict):
                    meta_keys = sorted(img_metas[0].keys())
                elif isinstance(img_metas[0], (list, tuple)) and img_metas[0]:
                    if isinstance(img_metas[0][-1], dict):
                        meta_keys = sorted(img_metas[0][-1].keys())
            print(
                '[MapTRPoseGatedV2I] phase={} gate_mode={} '
                'use_pose_metadata={} learnable_gate={} gate_mean={:.4f} '
                'gate_min={:.4f} gate_max={:.4f} gate_finite={} '
                'fallback_count={} constant_gate_value={} level0_shape={} '
                'ego_shape={} rsu_shape={} meta_keys={}'.format(
                    phase,
                    stats.get('gate_mode'),
                    stats.get('use_pose_metadata'),
                    stats.get('learnable_gate'),
                    stats['gate_mean'],
                    stats['gate_min'],
                    stats['gate_max'],
                    stats['gate_finite'],
                    stats['fallback_count'],
                    stats.get('constant_gate_value'),
                    stats['level0_shape'],
                    stats['ego_level0_shape'],
                    stats['rsu_level0_shape'],
                    meta_keys,
                )
            )
            self._pose_gate_debug_printed = True
        return img_feats

    def _debug_after_losses(self, losses):
        if (
            not self.debug_gate
            or not self._is_rank0()
            or self._pose_gate_loss_debug_printed
        ):
            return
        try:
            finite_values = {}
            for key, value in losses.items():
                if torch.is_tensor(value):
                    finite_values[key] = bool(torch.isfinite(value).all().detach().cpu())
                elif isinstance(value, (list, tuple)):
                    tensor_values = [item for item in value if torch.is_tensor(item)]
                    if tensor_values:
                        finite_values[key] = all(
                            bool(torch.isfinite(item).all().detach().cpu())
                            for item in tensor_values
                        )
            print(
                '[MapTRPoseGatedV2I] loss_keys={} loss_finite={}'.format(
                    sorted(losses.keys()),
                    finite_values,
                )
            )
        except Exception as exc:
            print('[MapTRPoseGatedV2I] loss debug skipped: {}'.format(exc))
        self._pose_gate_loss_debug_printed = True

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain historical BEV features with the same RSU gate as training."""
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, channels, height, width = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(
                bs * len_queue,
                num_cams,
                channels,
                height,
                width,
            )
            imgs_queue, _ = self._select_topk_views(
                imgs_queue,
                [item for metas in img_metas_list for item in metas],
                phase='history_input',
            )
            img_feats_list = self.extract_feat(
                img=imgs_queue,
                len_queue=len_queue,
            )
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_metas = self._slice_img_metas(
                    img_metas,
                    self._view_indices(num_cams),
                    num_cams,
                )
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                img_feats = self._apply_pose_gate(
                    img_feats,
                    img_metas,
                    phase='history',
                )
                prev_bev = self.pts_bbox_head(
                    img_feats,
                    None,
                    img_metas,
                    prev_bev,
                    only_bev=True,
                )
            self.train()
            return prev_bev

    @force_fp32(apply_to=('img', 'points', 'prev_bev'))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
    ):
        lidar_feat = None
        if self.modality == 'fusion':
            lidar_feat = self.extract_lidar_feat(points)

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = (
            self.obtain_history_bev(prev_img, prev_img_metas)
            if len_queue > 1
            else None
        )

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img, img_metas = self._select_topk_views(
            img,
            img_metas,
            phase='train_input',
        )
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        img_feats = self._apply_pose_gate(
            img_feats,
            img_metas,
            phase='train',
        )
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats,
            lidar_feat,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            prev_bev,
        )

        losses.update(losses_pts)
        self._debug_after_losses(losses)
        return losses

    def simple_test(
        self,
        img_metas,
        img=None,
        points=None,
        prev_bev=None,
        rescale=False,
        **kwargs
    ):
        lidar_feat = None
        if self.modality == 'fusion':
            lidar_feat = self.extract_lidar_feat(points)
        img, img_metas = self._select_topk_views(
            img,
            img_metas,
            phase='test_input',
        )
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        img_feats = self._apply_pose_gate(
            img_feats,
            img_metas,
            phase='test',
        )

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats,
            lidar_feat,
            img_metas,
            prev_bev,
            rescale=rescale,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
