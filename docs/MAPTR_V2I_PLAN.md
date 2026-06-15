# MapTR V2I Extension Plan

## Camera Modes

- Ego-only: six vehicle cameras, implemented now
- RSU-only: four infrastructure cameras, dataset/config work pending
- V2I: six vehicle plus four RSU cameras, model work pending

The placeholder camera groups live in
`integrations/maptr/configs/simv2i_maptr_v2i_placeholder.py`.

## Naive Ten-Camera Diagnostic

A flattened ten-camera input can check path loading, calibration, memory use,
and whether all views reach the encoder. It is not the target V2I model:

- Camera embeddings do not identify platform-specific uncertainty.
- RSU observations are not explicitly transformed into the ego BEV frame.
- Timestamp skew and unavailable RSU views have no dedicated handling.
- A single branch cannot clearly measure ego versus infrastructure gain.

This mode should be labeled `naive_flattened` and used only as a diagnostic.

## Proper V2I Design

1. Build separate ego and infrastructure image/BEV branches.
2. Transform RSU BEV features into the current ego frame using calibrated
   world poses.
3. Add timestamp matching, valid-view masks, and stale-message rejection.
4. Fuse aligned BEV features with masked attention or confidence gating.
5. Keep the MapTR vector head and GT in the ego coordinate frame.
6. Report ego-only, RSU-only, naive ten-camera, and proper V2I ablations.

Required additional pickle metadata:

- Camera timestamps and frame synchronization IDs
- RSU camera intrinsics and camera-to-RSU extrinsics
- RSU pose in world coordinates
- Ego pose in world coordinates
- Per-camera validity and communication availability

## Implementation Boundaries

This setup does not implement LiDAR fusion, an RSU encoder, BEV alignment,
communication simulation, or a V2I fusion module. The placeholder is
intentionally not trainable so it cannot be mistaken for the proper model.
