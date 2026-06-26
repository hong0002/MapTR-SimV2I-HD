#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path


CAMERA_NAMES = (
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
)
EXPECTED_SAMPLES = {
    'train': 7000,
    'val': 1000,
    'test': 2000,
}
WINDOWS_ABSOLUTE_RE = re.compile(r'^[A-Za-z]:[\\/]')


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description='Validate every SimV2I-HD benchmark v1.0 MapTR image path.'
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
        default=Path('outputs/maptr/ego_r18_stronger_10k_validation'),
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


def validate_split(root, dataset_dir, split):
    ann_path = dataset_dir / 'simv2i_maptr_infos_{}.pkl'.format(split)
    result = {
        'split': split,
        'ann_file': str(ann_path),
        'expected_samples': EXPECTED_SAMPLES[split],
        'sample_count': 0,
        'checked_image_count': 0,
        'missing_image_count': 0,
        'first_missing_paths': [],
        'errors': [],
    }
    if not ann_path.is_file():
        result['errors'].append('missing annotation file')
        result['ok'] = False
        return result

    try:
        infos, metadata = load_infos(ann_path)
    except Exception as exc:
        result['errors'].append('failed to load pickle: {!r}'.format(exc))
        result['ok'] = False
        return result

    result['sample_count'] = len(infos)
    result['metadata'] = metadata
    if len(infos) != EXPECTED_SAMPLES[split]:
        result['errors'].append(
            'expected {} samples, got {}'.format(
                EXPECTED_SAMPLES[split], len(infos)
            )
        )

    for sample_index, info in enumerate(infos):
        token = info.get('token', '<sample:{}>'.format(sample_index))
        cams = info.get('cams', {})
        if not isinstance(cams, dict):
            result['missing_image_count'] += len(CAMERA_NAMES)
            for camera_name in CAMERA_NAMES:
                append_limited(
                    result['first_missing_paths'],
                    {
                        'sample_index': sample_index,
                        'sample_token': token,
                        'camera': camera_name,
                        'path': None,
                        'reason': 'cams is not a dictionary',
                    },
                )
            continue

        for camera_name in CAMERA_NAMES:
            result['checked_image_count'] += 1
            cam_info = cams.get(camera_name)
            if not isinstance(cam_info, dict):
                result['missing_image_count'] += 1
                append_limited(
                    result['first_missing_paths'],
                    {
                        'sample_index': sample_index,
                        'sample_token': token,
                        'camera': camera_name,
                        'path': None,
                        'reason': 'missing camera entry',
                    },
                )
                continue

            data_path = cam_info.get('data_path')
            if not data_path:
                result['missing_image_count'] += 1
                append_limited(
                    result['first_missing_paths'],
                    {
                        'sample_index': sample_index,
                        'sample_token': token,
                        'camera': camera_name,
                        'path': data_path,
                        'reason': 'missing data_path',
                    },
                )
                continue

            resolved = resolve_path(root, data_path)
            if resolved is None or not resolved.is_file():
                result['missing_image_count'] += 1
                append_limited(
                    result['first_missing_paths'],
                    {
                        'sample_index': sample_index,
                        'sample_token': token,
                        'camera': camera_name,
                        'path': os.fspath(data_path),
                        'resolved_path': None if resolved is None else str(resolved),
                        'reason': (
                            'windows absolute path'
                            if resolved is None
                            else 'file does not exist'
                        ),
                    },
                )

    if result['missing_image_count']:
        result['errors'].append(
            'missing image paths: {}'.format(result['missing_image_count'])
        )
    expected_images = EXPECTED_SAMPLES[split] * len(CAMERA_NAMES)
    if result['checked_image_count'] != expected_images:
        result['errors'].append(
            'expected {} checked images, got {}'.format(
                expected_images, result['checked_image_count']
            )
        )
    result['ok'] = not result['errors']
    return result


def write_markdown(path, report):
    lines = [
        '# SimV2I-HD Benchmark v1.0 Image Path Validation',
        '',
        '- Status: `{}`'.format(report['status']),
        '- Project root: `{}`'.format(report['project_root']),
        '- Dataset directory: `{}`'.format(report['dataset_dir']),
        '',
        '| Split | Expected samples | Samples | Checked images | Missing images | Result |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for split in report['splits']:
        lines.append(
            '| {split} | {expected} | {samples} | {checked} | {missing} | {result} |'.format(
                split=split['split'],
                expected=split['expected_samples'],
                samples=split['sample_count'],
                checked=split['checked_image_count'],
                missing=split['missing_image_count'],
                result='PASS' if split['ok'] else 'FAIL',
            )
        )
    lines.append('')

    for split in report['splits']:
        lines.extend(['## {}'.format(split['split']), ''])
        if split['errors']:
            lines.append('Errors:')
            for error in split['errors']:
                lines.append('- {}'.format(error))
            lines.append('')
        if split['first_missing_paths']:
            lines.append('First missing paths:')
            for item in split['first_missing_paths']:
                lines.append('- `{}`'.format(item))
            lines.append('')
        else:
            lines.append('- First missing paths: `[]`')
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

    split_results = [
        validate_split(root, dataset_dir, split)
        for split in ('train', 'val', 'test')
    ]
    status = 'passed' if all(item['ok'] for item in split_results) else 'failed'
    report = {
        'status': status,
        'project_root': str(root),
        'dataset_dir': str(dataset_dir),
        'expected_camera_names': list(CAMERA_NAMES),
        'splits': split_results,
    }

    json_path = output_dir / 'image_path_validation.json'
    md_path = output_dir / 'image_path_validation.md'
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    write_markdown(md_path, report)

    print('SimV2I-HD benchmark 10k image path status: {}'.format(status))
    for split in split_results:
        print(
            '{split}: samples={sample_count}, checked_images={checked_image_count}, missing_images={missing_image_count}'.format(
                **split
            )
        )
    print('JSON report: {}'.format(json_path))
    print('Markdown report: {}'.format(md_path))
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
