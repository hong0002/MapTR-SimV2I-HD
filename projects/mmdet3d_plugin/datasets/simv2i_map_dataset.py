import copy
import os
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from pyquaternion import Quaternion
from shapely.geometry import LineString

from .nuscenes_dataset import CustomNuScenesDataset
from .nuscenes_map_dataset import (
    CustomNuScenesLocalMapDataset,
    LiDARInstanceLines,
)


EGO_CAMERA_NAMES = (
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
)
RSU_CAMERA_NAMES = (
    'rsu_00_rgb',
    'rsu_01_rgb',
    'rsu_02_rgb',
    'rsu_03_rgb',
)
V2I_VIEW_MODES = {
    'ego_only': ('ego', 0),
    'infra_only_4rsu': ('infra', 4),
    'v2i_1rsu': ('v2i', 1),
    'v2i_2rsu': ('v2i', 2),
    'v2i_4rsu': ('v2i', 4),
}


@DATASETS.register_module()
class SimV2IMapDataset(CustomNuScenesLocalMapDataset):
    """MapTR dataset for SimV2I-HD precomputed local-map vectors.

    The pickle follows the common ``{'infos': [...], 'metadata': {...}}``
    layout. Each info must contain ``cams`` and one of these vector layouts:

    - ``gt_vectors``/``vectors``: list of records with points and class name.
    - ``map_annos``/``gt_map_annos``: mapping from class name to polylines.
    - ``gt_vecs`` plus ``gt_labels``.

    Vector coordinates are already local ego coordinates: x right, y forward.
    No NuScenes map database or LiDAR input is used.
    """

    MAPCLASSES = ('divider', 'boundary', 'ped_crossing')

    def __init__(
        self,
        map_ann_file=None,
        camera_names=None,
        queue_length=1,
        bev_size=(80, 40),
        pc_range=(-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
        overlap_test=False,
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=None,
        view_mode='ego_only',
        rsu_camera_names=None,
        selected_cams=None,
        rsu_selection_policy='first_valid',
        *args,
        **kwargs
    ):
        self.map_ann_file = map_ann_file
        self.camera_names = tuple(camera_names or EGO_CAMERA_NAMES)
        self.rsu_camera_names = (
            tuple(RSU_CAMERA_NAMES)
            if rsu_camera_names is None
            else tuple(rsu_camera_names)
        )
        self.selected_cams = tuple(selected_cams or ())
        self.view_mode = view_mode
        self.rsu_selection_policy = rsu_selection_policy
        if self.selected_cams and self.view_mode != 'custom':
            self.view_mode = 'custom'
        if self.view_mode not in V2I_VIEW_MODES and self.view_mode != 'custom':
            raise ValueError(
                'Unsupported SimV2I view_mode: {}'.format(self.view_mode)
            )
        if self.rsu_selection_policy != 'first_valid':
            raise ValueError(
                'Unsupported RSU selection policy: {}'.format(
                    self.rsu_selection_policy
                )
            )
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = list(pc_range)
        self.patch_size = (
            self.pc_range[4] - self.pc_range[1],
            self.pc_range[3] - self.pc_range[0],
        )
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = (
            eval_use_same_gt_sample_num_flag
        )
        self.is_vis_on_test = False
        self.source_map_classes = None

        # Skip CustomNuScenesLocalMapDataset.__init__: it opens NuScenes maps.
        CustomNuScenesDataset.__init__(
            self,
            queue_length=queue_length,
            bev_size=bev_size,
            overlap_test=overlap_test,
            *args,
            **kwargs
        )
        if self.map_ann_file:
            map_ann_dir = osp.dirname(self.map_ann_file)
            if map_ann_dir:
                mmcv.mkdir_or_exist(map_ann_dir)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        if isinstance(data, dict) and 'infos' in data:
            infos = list(data['infos'])
            self.metadata = data.get('metadata', {})
            self.source_map_classes = (
                data.get('map_classes')
                or self.metadata.get('map_classes')
                or self.metadata.get('classes')
            )
        elif isinstance(data, list):
            infos = list(data)
            self.metadata = {}
        else:
            raise TypeError(
                'SimV2I annotation pickle must be a list or contain "infos".'
            )

        self.version = self.metadata.get('version', 'simv2i-hd-v1')
        infos.sort(key=lambda item: item.get('timestamp', 0))
        for index, info in enumerate(infos):
            info.setdefault('token', 'simv2i_{:08d}'.format(index))
            info.setdefault('scene_token', info.get('run_id', 'simv2i'))
            info.setdefault('frame_idx', index)
            info.setdefault('prev', '')
            info.setdefault('next', '')
            info.setdefault('sweeps', [])
        return infos[::self.load_interval]

    def get_cat_ids(self, idx):
        _, labels = self._extract_map_vectors(self.data_infos[idx])
        return sorted(set(labels))

    @staticmethod
    def _as_matrix(value, shape):
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.shape != shape:
            raise ValueError(
                'Expected matrix shape {}, got {}.'.format(shape, matrix.shape)
            )
        return matrix

    @staticmethod
    def _rotation_matrix(value):
        rotation = np.asarray(value, dtype=np.float64)
        if rotation.shape == (3, 3):
            return rotation.astype(np.float32)
        if rotation.size == 4:
            return Quaternion(rotation.tolist()).rotation_matrix.astype(
                np.float32
            )
        raise ValueError(
            'Rotation must be a 3x3 matrix or [w, x, y, z] quaternion.'
        )

    def _resolve_data_path(self, data_path):
        data_path = os.fspath(data_path)
        if osp.isabs(data_path) or (
            len(data_path) >= 3
            and data_path[1] == ':'
            and data_path[2] in ('/', '\\')
        ):
            return data_path
        normalized = osp.normpath(data_path)
        if not self.data_root or self.data_root == '.':
            return normalized
        root = osp.normpath(self.data_root)
        if normalized == root or normalized.startswith(root + osp.sep):
            return normalized
        return osp.join(root, normalized)

    def _camera_matrices(self, cam_info):
        intrinsic_value = cam_info.get(
            'cam_intrinsic', cam_info.get('intrinsics')
        )
        if intrinsic_value is None:
            raise KeyError('Camera entry is missing cam_intrinsic/intrinsics.')
        intrinsic = self._as_matrix(intrinsic_value, (3, 3))
        intrinsic_4x4 = np.eye(4, dtype=np.float32)
        intrinsic_4x4[:3, :3] = intrinsic

        camera2ego = None
        for key in ('camera2ego', 'cam2ego'):
            if key in cam_info:
                camera2ego = self._as_matrix(cam_info[key], (4, 4))
                break
        if camera2ego is None and 'sensor2ego_rotation' in cam_info:
            camera2ego = np.eye(4, dtype=np.float32)
            camera2ego[:3, :3] = self._rotation_matrix(
                cam_info['sensor2ego_rotation']
            )
            camera2ego[:3, 3] = np.asarray(
                cam_info.get('sensor2ego_translation', [0.0, 0.0, 0.0]),
                dtype=np.float32,
            )

        lidar2img = None
        for key in ('lidar2img', 'ego2img'):
            if key in cam_info:
                lidar2img = self._as_matrix(cam_info[key], (4, 4))
                break

        if lidar2img is None and 'sensor2lidar_rotation' in cam_info:
            sensor2lidar_rotation = self._as_matrix(
                cam_info['sensor2lidar_rotation'], (3, 3)
            )
            sensor2lidar_translation = np.asarray(
                cam_info['sensor2lidar_translation'], dtype=np.float32
            )
            lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
            lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
            lidar2cam = np.eye(4, dtype=np.float32)
            lidar2cam[:3, :3] = lidar2cam_r.T
            lidar2cam[3, :3] = -lidar2cam_t
            lidar2cam = lidar2cam.T
            lidar2img = intrinsic_4x4 @ lidar2cam
            if camera2ego is None:
                camera2ego = np.linalg.inv(lidar2cam).astype(np.float32)

        if lidar2img is None and 'lidar2cam' in cam_info:
            lidar2cam = self._as_matrix(cam_info['lidar2cam'], (4, 4))
            lidar2img = intrinsic_4x4 @ lidar2cam
            if camera2ego is None:
                camera2ego = np.linalg.inv(lidar2cam).astype(np.float32)

        if lidar2img is None and camera2ego is not None:
            lidar2img = intrinsic_4x4 @ np.linalg.inv(camera2ego)

        if lidar2img is None:
            raise KeyError(
                'Camera entry needs lidar2img/ego2img or usable extrinsics.'
            )
        if camera2ego is None:
            camera2ego = np.eye(4, dtype=np.float32)

        return (
            lidar2img.astype(np.float32),
            camera2ego.astype(np.float32),
            intrinsic_4x4,
        )

    @staticmethod
    def _has_valid_image(cam_info):
        return isinstance(cam_info, dict) and bool(
            cam_info.get('data_path') or cam_info.get('img_path')
        )

    def _rsu_items(self, info):
        rsu_cams = info.get('rsu_cams', {})
        if not isinstance(rsu_cams, dict):
            return []
        items = []
        selected_names = set()
        for name in self.rsu_camera_names:
            cam_info = rsu_cams.get(name)
            if self._has_valid_image(cam_info):
                items.append(('rsu', name, cam_info))
                selected_names.add(name)
        for name, cam_info in rsu_cams.items():
            if name in selected_names:
                continue
            if self._has_valid_image(cam_info):
                items.append(('rsu', name, cam_info))
                selected_names.add(name)
        return items

    def _ego_items(self, info):
        cams = info.get('cams', {})
        if not isinstance(cams, dict):
            raise KeyError(
                'Sample {} cams is not a dictionary.'.format(info['token'])
            )
        missing = [
            name
            for name in self.camera_names
            if not self._has_valid_image(cams.get(name))
        ]
        if missing:
            raise KeyError(
                'Sample {} is missing ego cameras: {}'.format(
                    info['token'], ', '.join(missing)
                )
            )
        return [('ego', name, cams[name]) for name in self.camera_names]

    def _custom_camera_items(self, info):
        cams = info.get('cams', {})
        rsu_cams = info.get('rsu_cams', {})
        items = []
        for raw_name in self.selected_cams:
            if raw_name.startswith('ego:'):
                group, name = 'ego', raw_name.split(':', 1)[1]
            elif raw_name.startswith('rsu:'):
                group, name = 'rsu', raw_name.split(':', 1)[1]
            elif raw_name in cams:
                group, name = 'ego', raw_name
            elif raw_name in rsu_cams:
                group, name = 'rsu', raw_name
            else:
                raise KeyError(
                    'Sample {} has no selected camera {}.'.format(
                        info['token'], raw_name
                    )
                )
            cam_info = cams[name] if group == 'ego' else rsu_cams[name]
            if not self._has_valid_image(cam_info):
                raise KeyError(
                    'Selected camera {} has no data_path in sample {}.'.format(
                        raw_name, info['token']
                    )
                )
            items.append((group, name, cam_info))
        return items

    def _selected_camera_items(self, info):
        if self.view_mode == 'custom':
            if not self.selected_cams:
                raise ValueError('custom view_mode requires selected_cams.')
            return self._custom_camera_items(info)

        mode_type, rsu_count = V2I_VIEW_MODES[self.view_mode]
        if mode_type == 'ego':
            return self._ego_items(info)

        rsu_items = self._rsu_items(info)
        if len(rsu_items) < rsu_count:
            raise KeyError(
                'Sample {} needs {} RSU cameras for {}, got {}.'.format(
                    info['token'], rsu_count, self.view_mode, len(rsu_items)
                )
            )
        rsu_items = rsu_items[:rsu_count]
        if mode_type == 'infra':
            return rsu_items
        return self._ego_items(info) + rsu_items

    def get_data_info(self, index):
        info = self.data_infos[index]
        ego_translation = np.asarray(
            info.get('ego2global_translation', [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        ego_rotation = info.get(
            'ego2global_rotation', [1.0, 0.0, 0.0, 0.0]
        )
        lidar_translation = np.asarray(
            info.get('lidar2ego_translation', [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        lidar_rotation = info.get(
            'lidar2ego_rotation', [1.0, 0.0, 0.0, 0.0]
        )
        lidar_path = info.get('lidar_path', '')

        can_bus = np.zeros(18, dtype=np.float32)
        raw_can_bus = np.asarray(info.get('can_bus', []), dtype=np.float32)
        can_bus[: min(len(raw_can_bus), 18)] = raw_can_bus[:18]
        can_bus[:3] = ego_translation
        can_bus[3:7] = np.asarray(ego_rotation, dtype=np.float32)
        yaw = Quaternion(ego_rotation).yaw_pitch_roll[0]
        can_bus[-2] = yaw
        can_bus[-1] = np.degrees(yaw)

        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = self._rotation_matrix(lidar_rotation)
        lidar2ego[:3, 3] = lidar_translation
        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, :3] = self._rotation_matrix(ego_rotation)
        ego2global[:3, 3] = ego_translation

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=self._resolve_data_path(lidar_path)
            if lidar_path
            else '',
            lidar_path=self._resolve_data_path(lidar_path)
            if lidar_path
            else '',
            sweeps=info.get('sweeps', []),
            ego2global_translation=ego_translation,
            ego2global_rotation=ego_rotation,
            lidar2ego_translation=lidar_translation,
            lidar2ego_rotation=lidar_rotation,
            prev_idx=info.get('prev', ''),
            next_idx=info.get('next', ''),
            scene_token=info['scene_token'],
            can_bus=can_bus,
            frame_idx=info['frame_idx'],
            timestamp=info.get('timestamp', index),
            lidar2ego=lidar2ego,
            lidar2global=ego2global @ lidar2ego,
            _raw_info=info,
        )

        if self.modality['use_camera']:
            selected_items = self._selected_camera_items(info)
            image_paths = []
            lidar2img = []
            camera2ego = []
            camera_intrinsics = []
            selected_camera_names = []
            selected_camera_groups = []
            for camera_group, camera_name, cam_info in selected_items:
                data_path = cam_info.get('data_path', cam_info.get('img_path'))
                if not data_path:
                    raise KeyError(
                        '{} has no data_path in sample {}.'.format(
                            camera_name, info['token']
                        )
                    )
                image_paths.append(self._resolve_data_path(data_path))
                cam_lidar2img, cam_camera2ego, cam_intrinsic = (
                    self._camera_matrices(cam_info)
                )
                lidar2img.append(cam_lidar2img)
                camera2ego.append(cam_camera2ego)
                camera_intrinsics.append(cam_intrinsic)
                selected_camera_names.append(camera_name)
                selected_camera_groups.append(camera_group)

            input_dict.update(
                img_filename=image_paths,
                lidar2img=lidar2img,
                camera2ego=camera2ego,
                camera_intrinsics=camera_intrinsics,
                selected_camera_names=selected_camera_names,
                selected_camera_groups=selected_camera_groups,
                view_mode=self.view_mode,
            )
        return input_dict

    def _label_from_value(self, value, source_classes=None):
        aliases = {
            'pedestrian_crossing': 'ped_crossing',
            'crosswalk': 'ped_crossing',
            'road_boundary': 'boundary',
        }
        if isinstance(value, str):
            class_name = aliases.get(value, value)
            if class_name not in self.MAPCLASSES:
                raise ValueError('Unsupported map class: {}'.format(value))
            return self.MAPCLASSES.index(class_name)

        label = int(value)
        if source_classes and 0 <= label < len(source_classes):
            return self._label_from_value(source_classes[label])
        if 0 <= label < len(self.MAPCLASSES):
            return label
        raise ValueError(
            'Map label {} is outside the configured classes.'.format(label)
        )

    @staticmethod
    def _points_from_record(record):
        if isinstance(record, dict):
            for key in ('pts', 'points', 'polyline', 'coords'):
                if key in record:
                    return record[key]
            raise KeyError('Vector record has no pts/points/polyline/coords.')
        return record

    def _append_vector(self, lines, labels, points, label):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            raise ValueError(
                'Map vector must have shape [N>=2, 2+], got {}.'.format(
                    points.shape
                )
            )
        points = points[:, :2]
        if not np.isfinite(points).all():
            raise ValueError('Map vector contains NaN or infinite coordinates.')
        lines.append(LineString(points))
        labels.append(label)

    def _extract_map_vectors(self, info):
        source_classes = (
            info.get('map_classes')
            or info.get('class_names')
            or self.source_map_classes
        )
        lines = []
        labels = []

        grouped = info.get('map_annos', info.get('gt_map_annos'))
        if grouped is not None:
            if not isinstance(grouped, dict):
                raise TypeError('map_annos/gt_map_annos must be a dictionary.')
            for class_value, records in grouped.items():
                label = self._label_from_value(class_value, source_classes)
                if (
                    isinstance(records, np.ndarray)
                    and records.ndim == 2
                    and records.shape[-1] >= 2
                ):
                    records = [records]
                for record in records:
                    self._append_vector(
                        lines, labels, self._points_from_record(record), label
                    )
            return lines, labels

        records = None
        for key in ('gt_vectors', 'vectors', 'map_vectors'):
            if key in info:
                records = info[key]
                break
        if records is not None:
            for record in records:
                if isinstance(record, dict):
                    class_value = None
                    for key in (
                        'cls_name',
                        'class_name',
                        'name',
                        'label',
                        'type',
                    ):
                        if key in record:
                            class_value = record[key]
                            break
                    if class_value is None:
                        raise KeyError('Vector record has no class name or label.')
                    points = self._points_from_record(record)
                elif isinstance(record, (tuple, list)) and len(record) == 2:
                    points, class_value = record
                else:
                    raise TypeError(
                        'Vector records must be dictionaries or (points, label).'
                    )
                label = self._label_from_value(class_value, source_classes)
                self._append_vector(lines, labels, points, label)
            return lines, labels

        points_key = None
        labels_key = None
        for candidate_points_key, candidate_labels_key in (
            ('gt_vecs', 'gt_labels'),
            ('maptr_gt_fixed_points', 'maptr_gt_labels'),
        ):
            if candidate_points_key in info and candidate_labels_key in info:
                points_key = candidate_points_key
                labels_key = candidate_labels_key
                break
        if points_key is not None:
            if len(info[points_key]) != len(info[labels_key]):
                raise ValueError(
                    '{} and {} lengths do not match.'.format(
                        points_key, labels_key
                    )
                )
            for points, class_value in zip(
                info[points_key], info[labels_key]
            ):
                label = self._label_from_value(class_value, source_classes)
                self._append_vector(lines, labels, points, label)
            return lines, labels

        raise KeyError(
            'No precomputed map vectors found. Expected gt_vectors, vectors, '
            'map_annos, gt_map_annos, gt_vecs plus gt_labels, or '
            'maptr_gt_fixed_points plus maptr_gt_labels.'
        )

    def vectormap_pipeline(self, example, input_dict):
        info = input_dict.get('_raw_info', input_dict)
        lines, labels = self._extract_map_vectors(info)
        instances = LiDARInstanceLines(
            lines,
            fixed_num=self.fixed_num,
            padding_value=self.padding_value,
            patch_size=self.patch_size,
        )
        example['gt_labels_3d'] = DC(
            torch.as_tensor(labels, dtype=torch.long), cpu_only=False
        )
        example['gt_bboxes_3d'] = DC(instances, cpu_only=True)
        return example

    def union2one(self, queue):
        # Parent implementation mutates can_bus; isolate the cached info data.
        return super().union2one(copy.deepcopy(queue))
