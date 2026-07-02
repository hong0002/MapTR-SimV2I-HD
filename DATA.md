# Data Layout

The paper configs use repo-relative paths. The safest public-release setup is
to keep the original config files unchanged and place or symlink the dataset
under `${MAPTR_ROOT}/data`.

## Environment Variables

```bash
export MAPTR_ROOT=/path/to/MapTR
export SIMV2I_HD_ROOT=/path/to/simv2i_hd_dataset
```

`SIMV2I_HD_ROOT` is a convenience variable for your local dataset storage. The
checked-in configs still read repo-relative paths such as `data/raw/...` and
`data/maptr/...`.

## Expected MapTR Annotation Layout

The main 20k dynamic-RSU split is expected at:

```text
${MAPTR_ROOT}/data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k/
├── simv2i_maptr_infos_train.pkl
├── simv2i_maptr_infos_val.pkl
├── simv2i_maptr_infos_test.pkl
├── simv2i_maptr_map_gt_train.json
├── simv2i_maptr_map_gt_val.json
└── simv2i_maptr_map_gt_test.json
```

Raw simulator images referenced by the PKLs should resolve from
`${MAPTR_ROOT}`, typically under:

```text
${MAPTR_ROOT}/data/raw/
```

## Using Symlinks For External Data

If the data is stored outside the repository, create symlinks into `data/`.
For example:

```bash
cd "${MAPTR_ROOT}"
mkdir -p data/maptr

ln -s "${SIMV2I_HD_ROOT}/raw" data/raw
ln -s "${SIMV2I_HD_ROOT}/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k" \
  data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k
```

Adjust the right-hand side to match your local storage. The important part is
that the config-visible path remains:

```text
data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k
```

## Validation

Run the lightweight validator before training or evaluation:

```bash
cd "${MAPTR_ROOT}"
bash scripts/validate_server_maptr_data.sh \
  --dataset-dir data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k \
  --require-data
```

The validator checks that the split PKLs exist, that vector-map classes are
recognized, and that referenced image paths are readable.

Reports are written under:

```text
outputs/maptr/server_setup/
```

This directory is intentionally ignored by git.

## Expected Map Classes

The HD map classes used by the MapTR configs are:

- `divider`
- `boundary`
- `ped_crossing`

The BEV range used by the paper visualizations is:

- x in `[-15, 15]`
- y in `[-30, 30]`
