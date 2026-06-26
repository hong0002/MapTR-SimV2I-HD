#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path


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
VIEW_MODES = {
    'ego_only': ('ego', 0),
    'infra_only_4rsu': ('infra', 4),
    'v2i_1rsu': ('v2i', 1),
    'v2i_2rsu': ('v2i', 2),
    'v2i_4rsu': ('v2i', 4),
}
EXPECTED_SAMPLES = {
    'train': 7000,
    'val': 1000,
    'test': 2000,
}
WINDOWS_ABSOLUTE_RE = re.compile(r'^[A-Za-z]:[\\/]')


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description='Validate SimV2I-HD V2I/RSU camera view modes.'
    )
    parser.add_argument('--root', type=Path, default=repo_root)
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('data/maptr/simv2i_hd_benchmark_v1'),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/maptr/v2i_fusion_precheck'),
    )
    parser.add_argument(
        '--view-mode',
        action='append',
        choices=sorted(VIEW_MODES),
        help='View mode to validate. Defaults to all required modes.',
    )
    return parser.parse_args()


def load_infos(path):
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and 'infos' in payload:
        return list(payload['infos']), payload.get('metadata', {})
    if isinstance(payload, list):
        return list(payload), {}
    raise TypeError(
        '{} must be a list or a dictionary containing "infos"'.format(path)
    )


def resolve_path(root, value):
    value = os.fspath(value)
    if WINDOWS_ABSOLUTE_RE.match(value):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def append_limited(items, item, limit=20):
    if len(items) < limit:
        items.append(item)


def has_valid_image(cam_info):
    return isinstance(cam_info, dict) and bool(
        cam_info.get('data_path') or cam_info.get('img_path')
    )


def selected_items(info, view_mode):
    cams = info.get('cams', {})
    rsu_cams = info.get('rsu_cams', {})
    if not isinstance(cams, dict):
        raise KeyError('cams is not a dictionary')
    if not isinstance(rsu_cams, dict):
        rsu_cams = {}

    def ego_items():
        items = []
        for name in EGO_CAMERA_NAMES:
            cam_info = cams.get(name)
            if not has_valid_image(cam_info):
                raise KeyError('missing ego camera {}'.format(name))
            items.append(('ego', name, cam_info))
        return items

    rsu_items = [
        ('rsu', name, rsu_cams[name])
        for name in RSU_CAMERA_NAMES
        if has_valid_image(rsu_cams.get(name))
    ]

    mode_type, rsu_count = VIEW_MODES[view_mode]
    if mode_type == 'ego':
        return ego_items()
    if len(rsu_items) < rsu_count:
        raise KeyError(
            'need {} RSU cameras, got {}'.format(rsu_count, len(rsu_items))
        )
    rsu_items = rsu_items[:rsu_count]
    if mode_type == 'infra':
        return rsu_items
    return ego_items() + rsu_items


def matrix_status(cam_info):
    has_intrinsic = 'cam_intrinsic' in cam_info or 'intrinsics' in cam_info
    has_lidar2img = 'lidar2img' in cam_info or 'ego2img' in cam_info
    has_sensor2ego = (
        'sensor2ego_rotation' in cam_info
        and 'sensor2ego_translation' in cam_info
    )
    has_sensor2lidar = (
        'sensor2lidar_rotation' in cam_info
        and 'sensor2lidar_translation' in cam_info
    )
    has_camera2ego = 'camera2ego' in cam_info or 'cam2ego' in cam_info
    usable_extrinsic = has_lidar2img or has_sensor2lidar or has_camera2ego
    return {
        'has_intrinsic': has_intrinsic,
        'has_lidar2img': has_lidar2img,
        'has_sensor2ego': has_sensor2ego,
        'has_sensor2lidar': has_sensor2lidar,
        'has_camera2ego': has_camera2ego,
        'usable_for_loader': has_intrinsic and usable_extrinsic,
    }


def inspect_image(path):
    try:
        from PIL import Image
    except ImportError as exc:
        return {'ok': False, 'error': 'Pillow import failed: {}'.format(exc)}
    try:
        with Image.open(path) as image:
            image.load()
            bands = image.getbands()
            return {
                'ok': True,
                'mode': image.mode,
                'channels': len(bands),
                'size': list(image.size),
                'loader_can_drop_alpha': len(bands) in (3, 4),
            }
    except Exception as exc:
        return {'ok': False, 'error': repr(exc)}


def validate_split(root, dataset_dir, split, view_mode):
    ann_file = dataset_dir / 'simv2i_maptr_infos_{}.pkl'.format(split)
    result = {
        'split': split,
        'view_mode': view_mode,
        'ann_file': str(ann_file),
        'sample_count': 0,
        'selected_camera_count_distribution': {},
        'checked_image_count': 0,
        'missing_image_count': 0,
        'matrix_missing_count': 0,
        'rgba_or_rgb_decode_count': 0,
        'first_missing_paths': [],
        'first_matrix_errors': [],
        'selected_camera_examples': [],
        'errors': [],
    }
    infos, metadata = load_infos(ann_file)
    result['sample_count'] = len(infos)
    result['metadata'] = metadata
    if len(infos) != EXPECTED_SAMPLES[split]:
        result['errors'].append(
            'expected {} samples, got {}'.format(
                EXPECTED_SAMPLES[split], len(infos)
            )
        )

    camera_count_hist = Counter()
    for sample_index, info in enumerate(infos):
        token = info.get('token', '<sample:{}>'.format(sample_index))
        try:
            items = selected_items(info, view_mode)
        except Exception as exc:
            result['errors'].append(
                'sample {} selection failed: {!r}'.format(sample_index, exc)
            )
            continue

        camera_count_hist[len(items)] += 1
        if len(result['selected_camera_examples']) < 5:
            result['selected_camera_examples'].append(
                {
                    'sample_index': sample_index,
                    'sample_token': token,
                    'cameras': [
                        '{}:{}'.format(group, name)
                        for group, name, _ in items
                    ],
                }
            )

        for group, name, cam_info in items:
            result['checked_image_count'] += 1
            data_path = cam_info.get('data_path', cam_info.get('img_path'))
            resolved = resolve_path(root, data_path)
            if resolved is None or not resolved.is_file():
                result['missing_image_count'] += 1
                append_limited(
                    result['first_missing_paths'],
                    {
                        'sample_index': sample_index,
                        'sample_token': token,
                        'camera': '{}:{}'.format(group, name),
                        'path': data_path,
                        'resolved_path': None if resolved is None else str(resolved),
                    },
                )
            elif sample_index == 0:
                image_status = inspect_image(resolved)
                if image_status.get('ok') and image_status.get(
                    'loader_can_drop_alpha'
                ):
                    result['rgba_or_rgb_decode_count'] += 1
                else:
                    result['errors'].append(
                        'first sample image decode failed for {}:{}: {}'.format(
                            group, name, image_status
                        )
                    )

            status = matrix_status(cam_info)
            if not status['usable_for_loader']:
                result['matrix_missing_count'] += 1
                append_limited(
                    result['first_matrix_errors'],
                    {
                        'sample_index': sample_index,
                        'sample_token': token,
                        'camera': '{}:{}'.format(group, name),
                        'status': status,
                    },
                )

    result['selected_camera_count_distribution'] = dict(camera_count_hist)
    if result['missing_image_count']:
        result['errors'].append(
            'missing image count {}'.format(result['missing_image_count'])
        )
    if result['matrix_missing_count']:
        result['errors'].append(
            'camera matrix missing count {}'.format(
                result['matrix_missing_count']
            )
        )
    result['ok'] = not result['errors']
    return result


def write_markdown(path, report):
    lines = [
        '# SimV2I-HD Benchmark v1.0 V2I View Validation',
        '',
        '- Status: `{}`'.format(report['status']),
        '- Project root: `{}`'.format(report['project_root']),
        '- Dataset directory: `{}`'.format(report['dataset_dir']),
        '- RSU selection policy: `first_valid`',
        '',
        '| View mode | Split | Samples | Camera count distribution | Checked images | Missing images | Matrix missing | Result |',
        '|---|---|---:|---|---:|---:|---:|---|',
    ]
    for mode in report['view_modes']:
        for split in mode['splits']:
            lines.append(
                '| {mode} | {split} | {samples} | `{dist}` | {checked} | {missing} | {matrix} | {result} |'.format(
                    mode=mode['view_mode'],
                    split=split['split'],
                    samples=split['sample_count'],
                    dist=split['selected_camera_count_distribution'],
                    checked=split['checked_image_count'],
                    missing=split['missing_image_count'],
                    matrix=split['matrix_missing_count'],
                    result='PASS' if split['ok'] else 'FAIL',
                )
            )
    lines.append('')
    for mode in report['view_modes']:
        lines.extend(['## {}'.format(mode['view_mode']), ''])
        lines.append(
            '- Policy: `{}`'.format(mode['policy'])
        )
        lines.append('')
        for split in mode['splits']:
            lines.append('### {}'.format(split['split']))
            lines.append('')
            lines.append(
                '- Selected camera examples: `{}`'.format(
                    split['selected_camera_examples'][:2]
                )
            )
            if split['errors']:
                lines.append('- Errors: `{}`'.format(split['errors']))
            else:
                lines.append('- Errors: `[]`')
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

    view_modes = args.view_mode or [
        'ego_only',
        'infra_only_4rsu',
        'v2i_1rsu',
        'v2i_2rsu',
        'v2i_4rsu',
    ]
    mode_results = []
    for view_mode in view_modes:
        split_results = [
            validate_split(root, dataset_dir, split, view_mode)
            for split in ('train', 'val', 'test')
        ]
        mode_results.append(
            {
                'view_mode': view_mode,
                'policy': (
                    'baseline ego cameras'
                    if view_mode == 'ego_only'
                    else 'first_valid RSU order, no privileged GT/coverage score'
                ),
                'splits': split_results,
                'ok': all(item['ok'] for item in split_results),
            }
        )
    report = {
        'status': (
            'passed' if all(item['ok'] for item in mode_results) else 'failed'
        ),
        'project_root': str(root),
        'dataset_dir': str(dataset_dir),
        'ego_camera_names': list(EGO_CAMERA_NAMES),
        'rsu_camera_names': list(RSU_CAMERA_NAMES),
        'view_modes': mode_results,
    }

    json_path = output_dir / 'view_validation.json'
    md_path = output_dir / 'view_validation.md'
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    write_markdown(md_path, report)

    print('SimV2I-HD V2I view validation status: {}'.format(report['status']))
    for mode in mode_results:
        for split in mode['splits']:
            print(
                '{mode} {split_label}: samples={sample_count}, camera_dist={dist}, checked_images={checked_image_count}, missing_images={missing_image_count}, matrix_missing={matrix_missing_count}'.format(
                    mode=mode['view_mode'],
                    split_label=split['split'],
                    dist=split['selected_camera_count_distribution'],
                    **split
                )
            )
    print('JSON report: {}'.format(json_path))
    print('Markdown report: {}'.format(md_path))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
