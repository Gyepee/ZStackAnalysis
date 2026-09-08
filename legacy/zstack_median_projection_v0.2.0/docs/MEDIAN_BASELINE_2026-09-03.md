# Median reconstruction baseline check

## 결정한 범위

- 각 Z plane의 60개 frame을 모두 사용
- frame별 registration 없음
- frame rejection 없음
- pixelwise median으로 plane image 생성
- mean projection을 primary figure로 생성
- maximum projection은 같은 intensity transform을 쓰는 비교 panel로만 생성
- Z/axial motion correction이라고 해석하지 않음

## 대표 실데이터 결과

입력: ROS-2335, 2026-09-01, `scan9G1ZQTYA`, 2 µm Z-step,
100 planes × 60 frames/plane

새 run:
`analysis_runs/zstack_median__20260903T112704Z__a36e066a/`

- 사용 frame: 6,000/6,000
- 제외 frame: 0
- page grouping 및 Z-spacing metadata gate: pass
- plane별 frame-to-median correlation 중앙값의 전체 중앙값: 0.1616
- descriptive low-correlation frame: 15/6,000; 최대 연속 2 frame
- median volume의 adjacent-plane NCC: mean 0.6161, median 0.6241

단일 raw frame의 SNR이 낮으므로 frame-to-median correlation의 절대값 자체를 motion
확률로 해석하지 않습니다. 낮은 correlation 표시는 검토용이며 aggregation에는
영향을 주지 않습니다.

## 기존 filtered mean과의 제한적 비교

같은 scan의 기존 correlation-filtered, unregistered mean volume과 비교했습니다.

| metric | 기존 filtered mean | 모든 frame pixelwise median |
|---|---:|---:|
| adjacent-plane NCC mean | 0.7573 | 0.6161 |
| adjacent-plane NCC median | 0.7861 | 0.6241 |
| single-slice XZ normalized Tenengrad | 0.1621 | 0.3529 |
| single-slice YZ normalized Tenengrad | 0.1299 | 0.2708 |

두 volume의 전체 voxel correlation은 0.6771이고, plane별 correlation 중앙값은
0.5792였습니다. 즉 median은 이 예에서 orthogonal edge metric을 높였지만
adjacent-plane similarity를 낮췄습니다. 따라서 “median이 전반적으로 더 안정적”이라고
아직 결론 내릴 수 없습니다. Median이 transient outlier에는 robust하다는 synthetic
test와, 실제 sequential two-photon signal에서 더 나은 reconstruction이라는 판단은
분리해야 합니다.

## 그림 읽는 법

- `fig_mean_projections.png`: mean projection 전용 표시 범위로 만든 primary view
- `fig_mean_vs_max_projections.png`: source volume에서 정한 하나의 표시 변환으로
  mean과 maximum projection을 직접 비교
- `fig_representative_median_plane.png`: projection이 아닌 중앙 Z plane의 median image
- `projection_source_arrays_float32.npz`: 표시 전 raw-float mean/max projection 배열

Maximum projection은 밝은 구조와 noise/extreme voxel을 강조하고, mean projection은
projection axis 전체의 평균을 보여 주므로 더 흐리지만 axial elongation의 과장을
줄여 확인할 수 있습니다.

## 남은 한계

Pixelwise median은 frame의 절반보다 적은 transient displacement/outlier에는 강하지만,
한 방향의 지속 drift, Z displacement, axial PSF, activity-dependent fluorescence,
photobleaching을 복원하지 않습니다. Z 안정성을 직접 판단하려면 반복 reference
Z-stack 또는 bead/phantom 같은 별도 기준이 필요합니다.
