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
        *args,
        **kwargs
    ):
        self.map_ann_file = map_ann_file
        self.camera_names = tuple(camera_names or EGO_CAMERA_NAMES)
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
            cams = info.get('cams', {})
            missing = [name for name in self.camera_names if name not in cams]
            if missing:
                raise KeyError(
                    'Sample {} is missing cameras: {}'.format(
                        info['token'], ', '.join(missing)
                    )
                )

            image_paths = []
            lidar2img = []
            camera2ego = []
            camera_intrinsics = []
            for camera_name in self.camera_names:
                cam_info = cams[camera_name]
                if 'data_path' not in cam_info:
                    raise KeyError(
                        '{} has no data_path in sample {}.'.format(
                            camera_name, info['token']
                        )
                    )
                image_paths.append(
                    self._resolve_data_path(cam_info['data_path'])
                )
                cam_lidar2img, cam_camera2ego, cam_intrinsic = (
                    self._camera_matrices(cam_info)
                )
                lidar2img.append(cam_lidar2img)
                camera2ego.append(cam_camera2ego)
                camera_intrinsics.append(cam_intrinsic)

            input_dict.update(
                img_filename=image_paths,
                lidar2img=lidar2img,
                camera2ego=camera2ego,
                camera_intrinsics=camera_intrinsics,
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

        if 'gt_vecs' in info and 'gt_labels' in info:
            if len(info['gt_vecs']) != len(info['gt_labels']):
                raise ValueError('gt_vecs and gt_labels lengths do not match.')
            for points, class_value in zip(
                info['gt_vecs'], info['gt_labels']
            ):
                label = self._label_from_value(class_value, source_classes)
                self._append_vector(lines, labels, points, label)
            return lines, labels

        raise KeyError(
            'No precomputed map vectors found. Expected gt_vectors, vectors, '
            'map_annos, gt_map_annos, or gt_vecs plus gt_labels.'
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
