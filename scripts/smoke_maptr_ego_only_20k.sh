#!/usr/bin/env bash
set -euo pipefail

# Run after: conda activate maptr_simv2i

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_ego_only_r18_20k_smoke.py"
WORK_DIR="${MAPTR_TRAIN_WORK_DIR:-outputs/maptr/ego_only_r18_20k_smoke}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${GPUS:-${MAPTR_GPUS:-1}}"
PORT="${PORT:-28620}"
mkdir -p "${LOG_DIR}"

maptr_check_runtime

echo "Running ego-only 20k config/dataloader precheck..."
"${PYTHON_BIN}" - <<'PY'
import json
import pickle
from pathlib import Path

from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset

import projects.mmdet3d_plugin  # noqa: F401

cfg_path = 'integrations/maptr/configs/simv2i_maptr_ego_only_r18_20k_smoke.py'
out_path = Path('outputs/maptr/ego_only_r18_20k_smoke/dataloader_smoke.json')
out_path.parent.mkdir(parents=True, exist_ok=True)

cfg = Config.fromfile(cfg_path)
expected_counts = {'train': 14000, 'val': 2000, 'test': 4000}
actual_counts = {}
first_100_missing = {}
for split in ('train', 'val', 'test'):
    ann_file = Path(getattr(cfg.data, split).ann_file)
    with ann_file.open('rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    actual_counts[split] = len(infos)
    missing = []
    for info in infos[:100]:
        for cam in cfg.ego_camera_names:
            path = info['cams'][cam].get('data_path') or info['cams'][cam].get('img_path')
            if not path or not Path(path).is_file():
                missing.append(path)
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break
    first_100_missing[split] = missing

dataset = build_dataset(cfg.data.train)
info0 = dataset.get_data_info(0)
image_paths = info0['img_filename']
selected_names = info0.get('selected_camera_names', [])
selected_groups = info0.get('selected_camera_groups', [])
assert len(image_paths) == 6, len(image_paths)
assert selected_groups == ['ego'] * 6, selected_groups
assert all(not name.startswith('rsu_') for name in selected_names), selected_names
assert actual_counts == expected_counts, actual_counts
assert not any(first_100_missing.values()), first_100_missing

example = dataset[0]
img = example['img'].data
loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    dist=False,
    shuffle=False,
)
batch = next(iter(loader))
batch_img = batch['img'].data[0]

result = {
    'status': 'passed',
    'config': cfg_path,
    'expected_split_counts': expected_counts,
    'actual_split_counts': actual_counts,
    'first_100_missing': first_100_missing,
    'dataset_len': len(dataset),
    'view_mode': info0.get('view_mode'),
    'selected_camera_names': selected_names,
    'selected_camera_groups': selected_groups,
    'input_view_count': len(image_paths),
    'rsu_view_used': any(group == 'rsu' for group in selected_groups),
    'single_sample_img_shape': list(img.shape),
    'first_batch_img_shape': list(batch_img.shape),
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
if [[ "${GPUS}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    tools/train.py "${CONFIG}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}" "${ROOT_LOG_FILE}"
else
  "${PYTHON_BIN}" tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}" "${ROOT_LOG_FILE}"
fi
