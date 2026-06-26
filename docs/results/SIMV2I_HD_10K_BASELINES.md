# SimV2I-HD Benchmark v1.0 10k Baselines

## Ego-Only MapTR R18 Scratch

| Dataset | Config | Checkpoint | Selection | Val mAP | Test mAP | Test divider AP | Test boundary AP | Test ped AP | Test mAP@1.5 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `simv2i_hd_benchmark_v1` | `integrations/maptr/configs/simv2i_maptr_ego_r18_stronger_10k.py` | `outputs/maptr/ego_r18_stronger_10k/epoch_20.pth` | best validation mAP | 0.0852 | 0.0671 | 0.0570 | 0.0483 | 0.0959 | 0.1100 |

Epoch 20 is the official representative checkpoint because it had the best validation mAP. Epoch 24 is not used as the representative baseline checkpoint.

`mAP` and `mAP@1.5` are separate metrics; `mAP@1.5` is the threshold-1.5 Chamfer AP summary.
