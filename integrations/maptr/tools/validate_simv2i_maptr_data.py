#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import pickletools
import re
import sys
from collections import Counter
from pathlib import Path


EXPECTED_CLASSES = ('divider', 'boundary', 'ped_crossing')
EGO_CAMERAS = (
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
)
SPLITS = ('train', 'val', 'test')
WINDOWS_ABSOLUTE_RE = re.compile(r'^[A-Za-z]:[\\/]')
INVALID_PATH_MARKER = '_invalid_pre_od_yflip'


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description='Validate SimV2I-HD MapTR pickle and image paths.'
    )
    parser.add_argument('--root', type=Path, default=repo_root)
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('data/maptr/simv2i_hd_v1'),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/maptr/server_setup'),
    )
    parser.add_argument(
        '--require-data',
        action='store_true',
        help='Return non-zero when any expected pickle is missing.',
    )
    return parser.parse_args()


def json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    return repr(value)


def detect_pickle_protocol(path):
    highest_protocol = 0
    with path.open('rb') as handle:
        for opcode, _, _ in pickletools.genops(handle):
            highest_protocol = max(highest_protocol, opcode.proto)
    return highest_protocol


def unpack_pickle(payload):
    if isinstance(payload, dict) and 'infos' in payload:
        infos = payload['infos']
        metadata = payload.get('metadata', {})
        declared_classes = (
            payload.get('map_classes')
            or metadata.get('map_classes')
            or metadata.get('classes')
        )
    elif isinstance(payload, list):
        infos = payload
        metadata = {}
        declared_classes = None
    else:
        raise TypeError(
            'pickle must be a list or a dictionary containing "infos"'
        )
    if not isinstance(infos, (list, tuple)):
        raise TypeError('"infos" must be a list or tuple')
    return list(infos), metadata, declared_classes


def class_name(value, declared_classes):
    aliases = {
        'pedestrian_crossing': 'ped_crossing',
        'crosswalk': 'ped_crossing',
        'road_boundary': 'boundary',
    }
    if isinstance(value, str):
        return aliases.get(value, value)
    try:
        label = int(value)
    except (TypeError, ValueError):
        return '<invalid:{}>'.format(value)
    if declared_classes and 0 <= label < len(declared_classes):
        return class_name(declared_classes[label], None)
    if 0 <= label < len(EXPECTED_CLASSES):
        return EXPECTED_CLASSES[label]
    return '<label:{}>'.format(label)


def vector_count_and_classes(info, declared_classes):
    observed = Counter()
    grouped = info.get('map_annos', info.get('gt_map_annos'))
    if grouped is not None:
        if not isinstance(grouped, dict):
            raise TypeError('map_annos/gt_map_annos must be a dictionary')
        count = 0
        for class_value, records in grouped.items():
            name = class_name(class_value, declared_classes)
            if hasattr(records, 'ndim') and records.ndim == 2:
                record_count = 1
            else:
                record_count = len(records)
            observed[name] += record_count
            count += record_count
        return count, observed

    records = None
    for key in ('gt_vectors', 'vectors', 'map_vectors'):
        if key in info:
            records = info[key]
            break
    if records is not None:
        for record in records:
            if isinstance(record, dict):
                value = None
                for key in (
                    'cls_name',
                    'class_name',
                    'name',
                    'label',
                    'type',
                ):
                    if key in record:
                        value = record[key]
                        break
                if value is None:
                    raise KeyError('vector record has no class name or label')
            elif isinstance(record, (tuple, list)) and len(record) == 2:
                value = record[1]
            else:
                raise TypeError('unsupported vector record layout')
            observed[class_name(value, declared_classes)] += 1
        return len(records), observed

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
            raise ValueError('{} and {} lengths differ'.format(
                points_key, labels_key
            ))
        for value in info[labels_key]:
            observed[class_name(value, declared_classes)] += 1
        return len(info[points_key]), observed

    raise KeyError('no supported precomputed map vector field found')


def resolve_path(root, value):
    value = os.fspath(value)
    if WINDOWS_ABSOLUTE_RE.match(value):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def inspect_image(path):
    try:
        from PIL import Image
    except ImportError as exc:
        return {'ok': False, 'error': 'Pillow is not installed: {}'.format(exc)}

    try:
        with Image.open(str(path)) as image:
            image.load()
            bands = image.getbands()
            return {
                'ok': True,
                'path': str(path),
                'mode': image.mode,
                'size': list(image.size),
                'channels': len(bands),
                'bands': list(bands),
            }
    except Exception as exc:
        return {'ok': False, 'path': str(path), 'error': repr(exc)}


def append_limited(items, value, limit=50):
    if len(items) < limit:
        items.append(value)


def validate_split(root, split, pickle_path):
    result = {
        'split': split,
        'pickle_path': str(pickle_path),
        'exists': pickle_path.is_file(),
        'errors': [],
        'warnings': [],
    }
    if not result['exists']:
        result['errors'].append('missing pickle')
        return result

    try:
        protocol = detect_pickle_protocol(pickle_path)
        result['pickle_protocol'] = protocol
        if protocol > 4:
            result['errors'].append(
                'pickle protocol {} exceeds Python 3.7 limit 4'.format(
                    protocol
                )
            )
    except Exception as exc:
        result['errors'].append(
            'could not inspect pickle protocol: {!r}'.format(exc)
        )

    try:
        with pickle_path.open('rb') as handle:
            payload = pickle.load(handle)
        infos, metadata, declared_classes = unpack_pickle(payload)
    except Exception as exc:
        result['errors'].append('pickle load failed: {!r}'.format(exc))
        return result

    result['pickle_load_ok'] = True
    result['num_infos'] = len(infos)
    result['metadata'] = json_value(metadata)
    result['declared_map_classes'] = json_value(declared_classes)
    if declared_classes is None:
        result['warnings'].append(
            'metadata has no map_classes; numeric labels assume configured order'
        )
    else:
        normalized = [class_name(name, None) for name in declared_classes]
        if set(normalized) != set(EXPECTED_CLASSES):
            result['errors'].append(
                'declared classes {} do not match {}'.format(
                    normalized, list(EXPECTED_CLASSES)
                )
            )

    if not infos:
        result['errors'].append('infos is empty')
        return result

    camera_count_histogram = Counter()
    missing_camera_names = Counter()
    missing_camera_paths = []
    missing_camera_calibration = []
    windows_paths = []
    invalid_marker_paths = []
    backslash_paths = []
    missing_lidar_paths = []
    gt_instances = 0
    gt_class_counts = Counter()
    gt_layout_errors = []

    for index, info in enumerate(infos):
        if not isinstance(info, dict):
            append_limited(
                gt_layout_errors,
                {'index': index, 'error': 'info is not a dictionary'},
            )
            continue

        cams = info.get('cams', {})
        if not isinstance(cams, dict):
            result['errors'].append(
                'sample {} cams is not a dictionary'.format(index)
            )
            continue
        camera_count_histogram[len(cams)] += 1
        for camera_name in EGO_CAMERAS:
            if camera_name not in cams:
                missing_camera_names[camera_name] += 1
                continue
            cam_info = cams[camera_name]
            if not isinstance(cam_info, dict):
                append_limited(
                    missing_camera_calibration,
                    {
                        'sample_index': index,
                        'camera': camera_name,
                        'error': 'camera entry is not a dictionary',
                    },
                )
                data_path = None
            else:
                data_path = cam_info.get('data_path')
                has_intrinsic = (
                    'cam_intrinsic' in cam_info or 'intrinsics' in cam_info
                )
                has_extrinsic = (
                    'lidar2img' in cam_info
                    or 'ego2img' in cam_info
                    or 'lidar2cam' in cam_info
                    or 'camera2ego' in cam_info
                    or 'cam2ego' in cam_info
                    or (
                        'sensor2lidar_rotation' in cam_info
                        and 'sensor2lidar_translation' in cam_info
                    )
                    or 'sensor2ego_rotation' in cam_info
                )
                if not has_intrinsic or not has_extrinsic:
                    append_limited(
                        missing_camera_calibration,
                        {
                            'sample_index': index,
                            'camera': camera_name,
                            'has_intrinsic': has_intrinsic,
                            'has_extrinsic': has_extrinsic,
                        },
                    )
            if not data_path:
                append_limited(
                    missing_camera_paths,
                    {
                        'sample_index': index,
                        'camera': camera_name,
                        'path': None,
                    },
                )
                continue
            data_path = os.fspath(data_path)
            if WINDOWS_ABSOLUTE_RE.match(data_path):
                append_limited(windows_paths, data_path)
                continue
            if INVALID_PATH_MARKER in data_path:
                append_limited(invalid_marker_paths, data_path)
            if '\\' in data_path:
                append_limited(backslash_paths, data_path)
            resolved = resolve_path(root, data_path)
            if resolved is None or not resolved.is_file():
                append_limited(
                    missing_camera_paths,
                    {
                        'sample_index': index,
                        'camera': camera_name,
                        'path': data_path,
                    },
                )

        lidar_path = info.get('lidar_path')
        if lidar_path:
            lidar_path = os.fspath(lidar_path)
            if WINDOWS_ABSOLUTE_RE.match(lidar_path):
                append_limited(windows_paths, lidar_path)
            elif INVALID_PATH_MARKER in lidar_path:
                append_limited(invalid_marker_paths, lidar_path)
            else:
                resolved = resolve_path(root, lidar_path)
                if resolved is None or not resolved.is_file():
                    append_limited(
                        missing_lidar_paths,
                        {'sample_index': index, 'path': lidar_path},
                    )

        sample_classes = (
            info.get('map_classes')
            or info.get('class_names')
            or declared_classes
        )
        try:
            count, observed = vector_count_and_classes(info, sample_classes)
            gt_instances += count
            gt_class_counts.update(observed)
        except Exception as exc:
            append_limited(
                gt_layout_errors,
                {'index': index, 'error': repr(exc)},
            )

    result['camera_count_histogram'] = dict(camera_count_histogram)
    result['required_ego_cameras'] = list(EGO_CAMERAS)
    result['missing_camera_names'] = dict(missing_camera_names)
    result['missing_camera_paths_count'] = len(missing_camera_paths)
    result['missing_camera_paths_examples'] = missing_camera_paths
    result['missing_camera_calibration_examples'] = (
        missing_camera_calibration
    )
    result['windows_absolute_paths_examples'] = windows_paths
    result['invalid_marker_paths_examples'] = invalid_marker_paths
    result['backslash_paths_examples'] = backslash_paths
    result['optional_missing_lidar_paths_examples'] = missing_lidar_paths
    result['gt_instance_count'] = gt_instances
    result['gt_class_counts'] = dict(gt_class_counts)
    result['gt_layout_errors'] = gt_layout_errors

    if missing_camera_names:
        result['errors'].append('one or more required ego cameras are missing')
    if missing_camera_paths:
        result['errors'].append('one or more camera data_path files are missing')
    if missing_camera_calibration:
        result['errors'].append(
            'one or more ego cameras have incomplete calibration'
        )
    if windows_paths:
        result['errors'].append('Windows absolute paths were found')
    if invalid_marker_paths:
        result['errors'].append(
            '{} paths were found'.format(INVALID_PATH_MARKER)
        )
    if backslash_paths:
        result['errors'].append('Windows path separators were found')
    if missing_lidar_paths:
        result['warnings'].append(
            'optional lidar_path values were present but files are missing'
        )
    if gt_layout_errors:
        result['errors'].append('one or more samples have invalid GT layout')
    if gt_instances == 0:
        result['errors'].append('GT instance count is zero')
    unknown_classes = sorted(set(gt_class_counts) - set(EXPECTED_CLASSES))
    missing_classes = sorted(set(EXPECTED_CLASSES) - set(gt_class_counts))
    if unknown_classes:
        result['errors'].append(
            'unknown GT classes: {}'.format(', '.join(unknown_classes))
        )
    if missing_classes:
        result['warnings'].append(
            'no GT instances observed for: {}'.format(', '.join(missing_classes))
        )

    first_cams = infos[0].get('cams', {})
    first_images = {}
    for camera_name in EGO_CAMERAS:
        cam_info = first_cams.get(camera_name, {})
        data_path = cam_info.get('data_path') if isinstance(cam_info, dict) else None
        if not data_path:
            first_images[camera_name] = {
                'ok': False,
                'error': 'missing data_path',
            }
            continue
        resolved = resolve_path(root, data_path)
        if resolved is None:
            first_images[camera_name] = {
                'ok': False,
                'error': 'Windows absolute path',
            }
        else:
            first_images[camera_name] = inspect_image(resolved)
    result['first_sample_images'] = first_images
    if not all(item.get('ok') for item in first_images.values()):
        result['errors'].append(
            'one or more first-sample ego images could not be decoded'
        )

    result['ok'] = not result['errors']
    return result


def write_markdown(path, report):
    lines = [
        '# SimV2I-HD MapTR Data Integrity Report',
        '',
        '- Status: `{}`'.format(report['status']),
        '- Project root: `{}`'.format(report['project_root']),
        '- Dataset directory: `{}`'.format(report['dataset_dir']),
        '',
    ]
    if report['status'] == 'missing_data':
        lines.extend(
            [
                'Expected pickle files are not present yet.',
                '',
                'Place them under:',
                '',
                '```text',
                report['dataset_dir'],
                '├── simv2i_maptr_infos_train.pkl',
                '├── simv2i_maptr_infos_val.pkl',
                '└── simv2i_maptr_infos_test.pkl',
                '```',
                '',
            ]
        )

    lines.extend(
        [
            '| Split | Exists | Protocol | Infos | GT instances | Result |',
            '|---|---:|---:|---:|---:|---|',
        ]
    )
    for split in report['splits']:
        lines.append(
            '| {split} | {exists} | {protocol} | {infos} | {gt} | {result} |'.format(
                split=split['split'],
                exists=split.get('exists', False),
                protocol=split.get('pickle_protocol', '-'),
                infos=split.get('num_infos', '-'),
                gt=split.get('gt_instance_count', '-'),
                result='PASS' if split.get('ok') else 'FAIL',
            )
        )
    lines.append('')

    for split in report['splits']:
        lines.extend(['## {}'.format(split['split']), ''])
        if split.get('errors'):
            lines.append('Errors:')
            for error in split['errors']:
                lines.append('- {}'.format(error))
            lines.append('')
        if split.get('warnings'):
            lines.append('Warnings:')
            for warning in split['warnings']:
                lines.append('- {}'.format(warning))
            lines.append('')
        if split.get('gt_class_counts') is not None:
            lines.append(
                '- GT class counts: `{}`'.format(split['gt_class_counts'])
            )
            lines.append(
                '- Camera count histogram: `{}`'.format(
                    split.get('camera_count_histogram', {})
                )
            )
            lines.append('')

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    root = args.root.resolve()
    dataset_dir = (
        args.dataset_dir
        if args.dataset_dir.is_absolute()
        else root / args.dataset_dir
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_results = []
    for split in SPLITS:
        pickle_path = dataset_dir / (
            'simv2i_maptr_infos_{}.pkl'.format(split)
        )
        split_results.append(
            validate_split(root, split, pickle_path)
        )

    missing = [item for item in split_results if not item['exists']]
    failed = [
        item
        for item in split_results
        if item['exists'] and not item.get('ok', False)
    ]
    if missing:
        status = 'missing_data'
    elif failed:
        status = 'failed'
    else:
        status = 'passed'

    report = {
        'status': status,
        'project_root': str(root),
        'dataset_dir': str(dataset_dir),
        'expected_pickle_protocol_max': 4,
        'coordinate_convention': {
            'x': 'ego right positive',
            'y': 'ego forward positive',
            'map_range_x': [-15, 15],
            'map_range_y': [-30, 30],
            'opendrive_to_carla': {
                'carla_x': 'od_x',
                'carla_y': '-od_y',
                'carla_yaw': '-od_heading',
            },
        },
        'splits': split_results,
    }
    json_path = output_dir / 'data_integrity_report.json'
    md_path = output_dir / 'data_integrity_report.md'
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    write_markdown(md_path, report)

    print('SimV2I-HD MapTR data status: {}'.format(status))
    print('Dataset directory: {}'.format(dataset_dir))
    print('JSON report: {}'.format(json_path))
    print('Markdown report: {}'.format(md_path))
    if status == 'missing_data':
        print(
            'Copy train/val/test pickle files to the dataset directory, then '
            'copy referenced data/raw image trees to the project root.'
        )
        return 2 if args.require_data else 0
    if status == 'failed':
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
