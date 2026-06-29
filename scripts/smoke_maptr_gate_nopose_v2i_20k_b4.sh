#!/usr/bin/env bash
set -euo pipefail

# Run after: conda activate maptr_simv2i

cd /data1/jihong/MapTR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_gate_nopose_v2i_r18_20k_b4_smoke.py"
WORK_DIR="${MAPTR_TRAIN_WORK_DIR:-outputs/maptr/gate_nopose_v2i_r18_20k_b4_smoke}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${GPUS:-4}"
PORT="${PORT:-28831}"
EXPECTED_GATE_MODE="nopose"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/smoke_$(maptr_timestamp).log"
ROOT_LOG_FILE="${WORK_DIR}/smoke.log"
exec > >(tee "${LOG_FILE}" "${ROOT_LOG_FILE}") 2>&1

maptr_check_runtime

echo "Running No-pose Learnable Gate V2I 20k batch4 config/dataloader/model precheck..."
SMOKE_GPUS="${GPUS}" \
SMOKE_CONFIG="${CONFIG}" \
SMOKE_WORK_DIR="${WORK_DIR}" \
EXPECTED_GATE_MODE="${EXPECTED_GATE_MODE}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
import pickle
from pathlib import Path

import torch
from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector

import projects.mmdet3d_plugin  # noqa: F401

cfg_path = os.environ['SMOKE_CONFIG']
work_dir = Path(os.environ['SMOKE_WORK_DIR'])
out_path = work_dir / 'dataloader_model_smoke.json'
expected_gate_mode = os.environ['EXPECTED_GATE_MODE']
out_path.parent.mkdir(parents=True, exist_ok=True)

cfg = Config.fromfile(cfg_path)
expected_counts = {'train': 14000, 'val': 2000, 'test': 4000}
actual_counts = {}
for split in ('train', 'val', 'test'):
    ann_file = Path(getattr(cfg.data, split).ann_file)
    with ann_file.open('rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    actual_counts[split] = len(infos)

dataset = build_dataset(cfg.data.train)
info0 = dataset.get_data_info(0)
image_paths = info0['img_filename']
selected_names = info0.get('selected_camera_names', [])
selected_groups = info0.get('selected_camera_groups', [])
ego_count = selected_groups.count('ego')
rsu_count = selected_groups.count('rsu')
missing_first_sample = [path for path in image_paths if not Path(path).is_file()]
assert len(image_paths) == 10, len(image_paths)
assert ego_count == 6, selected_groups
assert rsu_count == 4, selected_groups
assert actual_counts == expected_counts, actual_counts
assert not missing_first_sample, missing_first_sample

loader = build_dataloader(
    dataset,
    samples_per_gpu=cfg.data.samples_per_gpu,
    workers_per_gpu=0,
    dist=False,
    shuffle=False,
)
batch = next(iter(loader))
batch_img = batch['img'].data[0]
batch_metas = batch['img_metas'].data[0]
current_metas = []
for item in batch_metas:
    if isinstance(item, (list, tuple)):
        item = item[-1]
    if isinstance(item, dict) and 0 in item and isinstance(item[0], dict):
        item = item[0]
    current_metas.append(item)

gpus = int(os.environ.get('SMOKE_GPUS', '4'))
total_batch = cfg.data.samples_per_gpu * gpus
assert cfg.data.samples_per_gpu == 1, cfg.data.samples_per_gpu
assert total_batch == gpus, total_batch

model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'),
)
assert model.__class__.__name__ == 'MapTRPoseGatedV2I', model.__class__.__name__
assert model.ego_view_count == 6, model.ego_view_count
assert model.rsu_view_count == 4, model.rsu_view_count
assert model.input_rsu_view_count == 4, model.input_rsu_view_count
assert model.pose_gate.gate_mode == expected_gate_mode, model.pose_gate.gate_mode
assert model.pose_gate.use_pose_metadata is False
assert model.pose_gate.gate_mlp is None
assert tuple(model.pose_gate.rsu_slot_gate.shape) == (4,)
pose_gate_param_count = sum(p.numel() for p in model.pose_gate.parameters())
assert pose_gate_param_count == 4, pose_gate_param_count

selected_img, selected_metas = model._select_topk_views(
    batch_img,
    current_metas,
    phase='smoke_precheck',
)
assert list(batch_img.shape)[2] == 10, list(batch_img.shape)
assert list(selected_img.shape)[2] == 10, list(selected_img.shape)
meta0 = selected_metas[0]
metadata_counts = {}
for key in ('camera2ego', 'lidar2img', 'camera_intrinsics', 'filename'):
    value = meta0.get(key)
    if isinstance(value, (list, tuple)):
        metadata_counts[key] = len(value)
assert metadata_counts.get('camera2ego') == 10, metadata_counts
assert metadata_counts.get('lidar2img') == 10, metadata_counts

dummy_feat = torch.ones(1, 10, 256, 3, 5, requires_grad=True)
gated_feats, gate_stats = model.pose_gate(
    [dummy_feat],
    selected_metas,
    collect_stats=True,
)
loss = gated_feats[0].sum()
loss.backward()
ego_delta = float((gated_feats[0][:, :6] - dummy_feat[:, :6]).abs().max().detach())
rsu_delta = float((gated_feats[0][:, 6:] - dummy_feat[:, 6:]).abs().max().detach())
assert gated_feats[0].shape[1] == 10, gated_feats[0].shape
assert gate_stats['gate_mode'] == expected_gate_mode, gate_stats
assert gate_stats['use_pose_metadata'] is False, gate_stats
assert gate_stats['learnable_gate'] is True, gate_stats
assert gate_stats['pose_abs_mean'] is None, gate_stats
assert gate_stats['rsu_count'] == 4, gate_stats
assert gate_stats['gate_finite'], gate_stats
assert gate_stats['fallback_count'] == 0, gate_stats
assert abs(gate_stats['gate_mean'] - 0.5) < 1e-6, gate_stats
assert ego_delta == 0.0, ego_delta
assert rsu_delta > 0.0, rsu_delta
assert bool(torch.isfinite(dummy_feat.grad).all()), 'non-finite dummy grad'
assert bool(torch.isfinite(model.pose_gate.rsu_slot_gate.grad).all())

metadata_check_feat = torch.ones(1, 10, 256, 3, 5)
with torch.no_grad():
    gated_with_meta, _ = model.pose_gate([metadata_check_feat], selected_metas)
    gated_without_meta, _ = model.pose_gate([metadata_check_feat], [{}])
metadata_independence_delta = float(
    (gated_with_meta[0] - gated_without_meta[0]).abs().max()
)
assert metadata_independence_delta < 1e-7, metadata_independence_delta

result = {
    'status': 'passed',
    'config': cfg_path,
    'model_class': model.__class__.__name__,
    'expected_split_counts': expected_counts,
    'actual_split_counts': actual_counts,
    'dataset_len': len(dataset),
    'view_mode': info0.get('view_mode'),
    'selected_camera_names': selected_names,
    'selected_camera_groups': selected_groups,
    'input_view_count': len(image_paths),
    'used_input_view_count': int(selected_img.shape[2]),
    'ego_view_count': ego_count,
    'rsu_view_count': rsu_count,
    'metadata_view_counts': metadata_counts,
    'samples_per_gpu': cfg.data.samples_per_gpu,
    'gpus': gpus,
    'total_batch': total_batch,
    'first_batch_img_shape': list(batch_img.shape),
    'first_batch_selected_img_shape': list(selected_img.shape),
    'first_sample_missing_images': missing_first_sample,
    'gate_stats': gate_stats,
    'pose_gate_param_count': pose_gate_param_count,
    'ego_delta': ego_delta,
    'rsu_delta': rsu_delta,
    'metadata_independence_delta': metadata_independence_delta,
}
out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, indent=2, sort_keys=True))
PY

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS - 1))")"
  export CUDA_VISIBLE_DEVICES
fi

echo "Using GPUS=${GPUS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  tools/train.py "${CONFIG}" \
  --launcher pytorch \
  --work-dir "${WORK_DIR}" \
  "$@"
