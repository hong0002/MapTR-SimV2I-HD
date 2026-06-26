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
        self.debug_gate = bool(debug_gate)
        self._pose_gate_debug_printed = False
        self.latest_pose_gate_stats = None

    def _apply_pose_gate(self, img_feats, img_metas, phase):
        collect_stats = self.debug_gate and not self._pose_gate_debug_printed
        img_feats, stats = self.pose_gate(
            img_feats,
            img_metas,
            collect_stats=collect_stats,
        )
        if stats is not None:
            self.latest_pose_gate_stats = stats
            print(
                '[MapTRPoseGatedV2I] phase={} gate_mean={:.4f} '
                'gate_min={:.4f} gate_max={:.4f} level0_shape={} '
                'ego_shape={} rsu_shape={}'.format(
                    phase,
                    stats['gate_mean'],
                    stats['gate_min'],
                    stats['gate_max'],
                    stats['level0_shape'],
                    stats['ego_level0_shape'],
                    stats['rsu_level0_shape'],
                )
            )
            self._pose_gate_debug_printed = True
        return img_feats

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
            img_feats_list = self.extract_feat(
                img=imgs_queue,
                len_queue=len_queue,
            )
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
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
