#!/usr/bin/env bash
set -euo pipefail

# Run after: conda activate maptr_simv2i

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_pose_gated_v2i_r18_20k_b4_smoke.py"
WORK_DIR="${MAPTR_TRAIN_WORK_DIR:-outputs/maptr/pose_gated_v2i_r18_20k_b4_smoke}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${GPUS:-${MAPTR_GPUS:-4}}"
PORT="${PORT:-28825}"
mkdir -p "${LOG_DIR}"

maptr_check_runtime

echo "Running pose-gated V2I 20k batch4 config/dataloader/model precheck..."
SMOKE_GPUS="${GPUS}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import pickle
from pathlib import Path

from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector

import projects.mmdet3d_plugin  # noqa: F401

cfg_path = (
    'integrations/maptr/configs/'
    'simv2i_maptr_pose_gated_v2i_r18_20k_b4_smoke.py'
)
out_path = Path(
    'outputs/maptr/pose_gated_v2i_r18_20k_b4_smoke/'
    'dataloader_model_smoke.json'
)
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
assert any(name.startswith('rsu_') for name in selected_names), selected_names
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
assert hasattr(model, 'pose_gate')

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
    'ego_view_count': ego_count,
    'rsu_view_count': rsu_count,
    'samples_per_gpu': cfg.data.samples_per_gpu,
    'gpus': gpus,
    'total_batch': total_batch,
    'first_batch_img_shape': list(batch_img.shape),
    'first_sample_missing_images': missing_first_sample,
}
out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, indent=2, sort_keys=True))
PY

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS - 1))")"
  export CUDA_VISIBLE_DEVICES
fi

echo "Using GPUS=${GPUS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

LOG_FILE="${LOG_DIR}/smoke_$(maptr_timestamp).log"
ROOT_LOG_FILE="${WORK_DIR}/smoke.log"
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  tools/train.py "${CONFIG}" \
  --launcher pytorch \
  --work-dir "${WORK_DIR}" \
  "$@" 2>&1 | tee "${LOG_FILE}" "${ROOT_LOG_FILE}"
