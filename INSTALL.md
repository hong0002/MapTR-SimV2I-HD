# Installation

This repository uses the legacy MapTR/OpenMMLab 1.x stack. The recommended path
is to use the checked-in conda environment file and then build the vendored
MMDetection3D and MapTR CUDA extensions.

## 1. Clone And Enter The Repository

```bash
export MAPTR_ROOT=/path/to/MapTR
cd "${MAPTR_ROOT}"
```

Optional helper environment variables used by scripts:

```bash
export MAPTR_PYTHON=python
export SIMV2I_HD_ROOT=/path/to/simv2i_hd_dataset
```

`MAPTR_PYTHON` can point to an explicit interpreter such as
`/path/to/miniconda/envs/maptr_simv2i/bin/python`.

## 2. Create The Conda Environment

```bash
conda env create -f integrations/maptr/environment/maptr_legacy.yml
conda activate maptr_simv2i
```

The reference environment uses:

- Python 3.8
- PyTorch 1.9.1
- CUDA toolkit 11.1
- MMCV-full 1.4.0
- MMDetection 2.14.0
- MMSegmentation 0.14.1
- bundled MMDetection3D 0.17.2 source

If `mmcv-full` needs to be installed manually:

```bash
pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1
```

## 3. Build The Bundled MMDetection3D Source

```bash
cd "${MAPTR_ROOT}/mmdetection3d"
python setup.py develop
```

## 4. Build The MapTR CUDA Operator

```bash
cd "${MAPTR_ROOT}/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn"
python setup.py build install
```

## 5. Verify Imports

```bash
cd "${MAPTR_ROOT}"
PYTHONPATH="${MAPTR_ROOT}:${MAPTR_ROOT}/mmdetection3d" "${MAPTR_PYTHON:-python}" -c \
  "import torch, mmcv, mmdet, mmdet3d; print(torch.__version__, mmcv.__version__, mmdet.__version__, mmdet3d.__version__, torch.cuda.is_available())"
```

## Notes

- Prefer the conda environment file for full setup.
- `requirements.txt` records the pip-side package constraints and is mainly a
  reference for reproducibility.
- Do not commit local datasets, checkpoints, outputs, or prediction PKLs.
