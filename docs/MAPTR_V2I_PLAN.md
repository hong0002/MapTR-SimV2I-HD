# MapTR V2I Design Notes

This note records the V2I design intent behind the SimV2I-HD MapTR integration.
The executable paper configs live under `integrations/maptr/configs/`.

## Camera Modes

- Ego-only: six vehicle cameras.
- Dynamic Top-4 RSU: selected infrastructure cameras are added using the
  dynamic RSU selection metadata.
- Pose-gated Top-4 V2I: selected RSU features are modulated by pose-aware gates
  before fusion.

## Diagnostic Ten-camera Input

A flattened ten-camera input can check path loading, calibration, memory use,
and whether all views reach the encoder. It is not the target V2I model:

- Camera embeddings do not identify platform-specific uncertainty.
- RSU observations are not explicitly transformed into the ego BEV frame.
- Timestamp skew and unavailable RSU views have no dedicated handling.
- A single branch cannot clearly measure ego versus infrastructure gain.

This mode should be labeled `naive_flattened` and used only as a diagnostic.

## V2I Design

1. Use ego and infrastructure image/BEV features with calibrated metadata.
2. Transform or align RSU observations into the current ego frame using world
   poses.
3. Use timestamp matching and valid-view masks when available.
4. Fuse aligned BEV features with confidence-aware or pose-aware gating.
5. Keep the MapTR vector head and GT in the ego coordinate frame.
6. Report ego-only, dynamic RSU selection, and pose-gated V2I ablations.

Required metadata:

- Camera timestamps and frame synchronization IDs
- RSU camera intrinsics and camera-to-RSU extrinsics
- RSU pose in world coordinates
- Ego pose in world coordinates
- Per-camera validity and communication availability

## Reporting Boundary

Main paper quantitative claims should use the scenario-disjoint controlled 20k
split. The geographically buffered subset is a diagnostic stress protocol for
studying held-out geographic-region generalization, not a replacement for the
main benchmark.
