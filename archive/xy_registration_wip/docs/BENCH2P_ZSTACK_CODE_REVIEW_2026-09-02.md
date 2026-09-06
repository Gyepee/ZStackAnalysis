# Bench2p Z-stack 코드·산출물 검토 handoff

검토일: 2026-09-02

범위: `datasets/analysis_runs/bench2p_zstack*`, 관련 workflow/config/test

목적: 1 µm와 2 µm Z-step 처리 여부, 현재 플롯의 실제 입력과 변환, Z-motion artifact 개선을 위한 문헌 조사 질문을 한 문서로 정리

## 한눈에 보는 결론

1. **Z-step은 설정 파일에 고정되어 있지 않고, 원본 TIFF의 ScanImage metadata에서 자동으로 읽힌다.**
   - `SI.hStackManager.actualStackZStepSize`를 `z_step_um`으로 읽는다.
   - `SI.hStackManager.zs`를 실제 plane별 `z_positions_um`으로 읽는다.
   - 결과 OME-TIFF의 `PhysicalSizeZ`에 Z-step을 기록하고, 후속 side-view 코드는 그 OME metadata를 다시 읽는다.
2. 실제 run manifest에도 **1 µm와 2 µm가 올바르게 분리되어 기록**되어 있다.
   - 2026-08-27 자료: `z_step_um = 1.0`
   - 2026-09-01 자료: `z_step_um = 2.0`
3. 물리적 aspect ratio와 µm 단위 Gaussian sigma/ROI/slab 두께에는 이 차이가 반영된다. 따라서 2 µm stack이 1 µm처럼 납작하게 취급되지는 않는다.
4. 그러나 현재 pipeline에는 **축방향(Z) motion correction이 없다.** 현재 구현은 다음뿐이다.
   - plane 안의 초기 5-frame 평균과 후기 5-frame 평균 사이 **XY rigid shift를 QC로 측정**
   - single-stack workflow에서는 median frame과 상관이 낮은 frame을 **제외한 뒤 평균**하되, frame 자체를 정렬하지는 않음
   - 두 stack을 잇는 workflow에서 경계 plane을 이용해 두 번째 stack 전체에 하나의 **XY rigid shift** 적용
5. 따라서 X-Z/Y-Z에서 보이는 Z 방향 늘어짐은 세포 형태만으로 해석할 수 없다. **axial PSF, 1/2 µm sampling, 순차 촬영 중 Z/XY motion, MIP, display transform**이 섞여 있다.
6. 1 µm와 2 µm 결과의 차이는 Z-step만의 차이가 아니다. 현재 자료에서는 frames/plane, 총 촬영 시간, XY pixel size, FOV, 날짜와 동물이 함께 달라졌다.

## 코드와 데이터의 실제 흐름

```text
raw ScanImage TIFF
  ├─ FrameData metadata
  │    ├─ actualNumSlices
  │    ├─ framesPerSlice
  │    ├─ zs
  │    ├─ actualStackZStepSize
  │    └─ imagingFovUm / scanFrameRate
  │
  ├─ 각 Z plane의 연속 frame 묶음
  │    ├─ two-stack: 모든 선택 frame의 arithmetic mean
  │    └─ single-stack: 선택적으로 low-correlation frame 제외 후 mean
  │
  └─ float32 Z-Y-X mean volume
       ├─ source_slice_metrics.csv
       ├─ OME-TIFF (PhysicalSizeZ/Y/X)
       ├─ axial/side/oblique projection figure
       └─ sideview workflow
            ├─ global percentile + asinh display transform
            ├─ physical cuboid alpha composite
            ├─ X-Z thin-slab MIP
            └─ bright-candidate XY/X-Z/Y-Z/local-oblique panels
```

주요 파일:

- metadata parsing, 평균, 두 stack merge, 기본 projection: `src/labgraph_ops/workflows/bench2p_zstack.py`
- 두 인접 stack 실행/manifest: `src/labgraph_ops/workflows/bench2p_zstack_run.py`
- 한 개 full-range stack 및 frame-stability filter: `src/labgraph_ops/workflows/bench2p_zstack_single.py`
- OME spacing 기반 cuboid/thin slab/candidate figure: `src/labgraph_ops/workflows/bench2p_zstack_sideview.py`
- 실행 진입점: `scripts/build_bench2p_zstack.py`, `scripts/build_bench2p_zstack_single.py`, `scripts/render_bench2p_zstack_sideview.py`
- 기본 설정: `config/bench2p_zstack.json`, `config/bench2p_zstack_single.json`
- 1 µm side-view 설정: `config/bench2p_zstack_sideview.json`
- 현재 2 µm cuboid 설정: `config/bench2p_zstack_cuboid.json`

운영상 주의: `render_bench2p_zstack_sideview.py`를 `--config` 없이 실행하면
`bench2p_zstack_sideview.json`, 즉 1 µm의 2026-08-28 run을 다시 사용한다. 현재
2 µm cuboid를 재현하려면 `--config config/bench2p_zstack_cuboid.json`을 명시해야 한다.

## Z-step metadata는 어디서 와서 어디에 쓰이는가

| 단계 | metadata/값 | 코드 위치 | 실제 사용 |
|---|---|---|---|
| TIFF 읽기 | `SI.hStackManager.actualStackZStepSize` | `bench2p_zstack.py:242-245` | `StackMetadata.z_step_um` |
| TIFF 읽기 | `SI.hStackManager.zs` | `bench2p_zstack.py:201-207` | plane별 절대/장비 Z 위치 목록 |
| TIFF 읽기 | `actualNumSlices`, `framesPerSlice` | `bench2p_zstack.py:193-200` | page grouping 및 page-count gate |
| XY calibration | `imagingFovUm` | `bench2p_zstack.py:209-213, 247-250` | `pixel_size_x/y_um = FOV / pixels` |
| source volume 저장 | `z_step_um` | `bench2p_zstack.py:491-514` | OME `PhysicalSizeZ` |
| 두 stack 병합 | `median(diff(z_positions))` | `bench2p_zstack_run.py:130-148` | merged OME의 scalar Z spacing |
| 기본 side/oblique | `z_step / mean(XY pixel size)` | `bench2p_zstack.py:570-650` | Z축을 XY scale에 맞춰 보간한 뒤 projection |
| 후속 sideview | OME `PhysicalSizeZ/Y/X` | `bench2p_zstack_sideview.py:66-88, 898-905` | 모든 물리 좌표와 resampling의 출발점 |
| feature detector | µm sigma / spacing | `bench2p_zstack_sideview.py:126-177` | 1/2 µm Z-step에 맞춰 Gaussian sigma를 voxel로 변환 |
| ROI/slab | µm size / spacing | `bench2p_zstack_sideview.py:180-205, 619-749` | ROI 및 slab을 가능한 한 비슷한 물리 크기로 선택 |

즉, **1 µm와 2 µm를 metadata에서 받아 처리하는 기본 경로는 구현되어 있다.** Z-step을 config에서 수동으로 1 또는 2로 지정하는 구조가 아니다.

다만 다음 검증은 아직 없다.

- `actualStackZStepSize`와 모든 `diff(z_positions_um)`가 실제로 일치하는지 확인하지 않는다.
- Z 위치가 엄격히 monotonic인지, 방향이 뒤집혔는지, 중간 plane이 빠졌는지 확인하지 않는다.
- single-stack admission gate는 `setup/enable/slow`를 보지만 Z-spacing의 양수성·규칙성은 검사하지 않는다.
- OME-TIFF에는 scalar spacing만 저장된다. 원래의 Z 시작점과 작은 irregularity는 `sources.json`/CSV에는 남지만 OME 좌표계에는 남지 않는다.
- sideview의 `candidate.z_um = z_index × spacing`은 **volume 시작점 기준 상대 좌표**이다. 예를 들어 `scan9G1ZQTYA`의 실제 첫 Z는 0.992 µm이므로 candidate table의 `z_um=38.0`은 ScanImage 좌표로는 약 38.992 µm에 해당한다. 현재 열 이름만 보면 이 차이가 드러나지 않는다.

## 확인된 1 µm와 2 µm acquisition

중복 재실행 run이 있으므로 아래 표는 대표 run 또는 고유 scan 기준이다.

| 날짜/동물/scan | Z-step | planes × frames/plane | Z center 범위 | XY pixel | 예상 순차 촬영 시간 | current QC |
|---|---:|---:|---:|---:|---:|---|
| 2026-08-27 ROS-2335 `scan9G1WRJ8Y` + `scan9G1WRM40` | 1 µm | 51×100 + 51×100 | 0.797–100.803 µm, 경계 1 plane 중복 | 1.2139 µm | 약 170.1 s/stack, 합계 약 340.3 s + stack 사이 간격 | pass, drift p95 2.137 px |
| 2026-08-27 ROS-2338 `scan9G1WRZL6` | 1 µm | 100×100 | 0–99 µm | 1.2139 µm | 약 333.7 s | pass, p95 1.972 px |
| 2026-08-27 ROS-2338 `scan9G1WRV2H` | 1 µm | 100×100 | 0.992–99.992 µm | 1.2139 µm | 약 333.7 s | needs_review, p95 25.852 px |
| 2026-09-01 ROS-2335 `scan9G1ZQQ9J` | 2 µm | 100×60 | 0–198 µm | 0.9104 µm | 약 200.1 s | needs_review, p95 3.265 px, max 13.744 px |
| 2026-09-01 ROS-2335 `scan9G1ZQTYA` | 2 µm | 100×60 | 0.992–198.992 µm | 0.9104 µm | 약 200.1 s | pass, p95 2.389 px, max 4.140 px |
| 2026-09-01 ROS-2335 `scan9G1ZQXUT` | 2 µm | 100×60 | 25.797–223.797 µm | 0.9104 µm | 약 200.1 s | pass, p95 2.120 px, max 3.162 px |
| 2026-09-01 ROS-2338 `scan9G1ZR53G` | 2 µm | 100×60 | 48.915–246.915 µm | 0.9104 µm | 약 200.1 s | needs_review, p95 3.617 px, max 5.224 px |

대표 provenance:

- 1 µm two-stack: `datasets/analysis_runs/bench2p_zstack__20260828T121243Z__32bc5a6f/`
- 1 µm single-stack/current filter: `datasets/analysis_runs/bench2p_zstack_single__20260828T142125Z__3083832d/`
- 2 µm source volumes: `datasets/analysis_runs/bench2p_zstack_single__20260901T173732Z__44d116a2/`, `...__8d89c756/`, `...__f87e2070/`, `bench2p_zstack_single__20260901T173733Z__62924206/`
- 최신 2 µm cuboid/sideview: `datasets/analysis_runs/bench2p_zstack_sideview__20260902T084750Z__7f5a5ed3/`

주의할 confound:

- 1 µm single stack은 100 frames/plane, 약 333.7초이고 2 µm stack은 60 frames/plane, 약 200.1초이다.
- XY sampling도 1.2139 µm와 0.9104 µm로 다르다.
- 따라서 관찰되는 artifact 차이는 Z-step, plane dwell time, 전체 stack duration, FOV/XY resolution, SNR, 날짜/동물의 조합이다.

## 현재 volume 생성과 motion 관련 함수

### 1. `mean_stack`: 두 stack workflow의 plane 평균

위치: `bench2p_zstack.py:335-410`

- 각 Z plane의 `frames_per_slice` frames를 읽는다.
- 현재 config에서는 초기 frame을 버리지 않고 전부 arithmetic mean한다.
- 처음 5 frames의 평균과 마지막 5 frames의 평균 사이의 XY shift를 phase cross-correlation으로 구한다.
- 그 shift는 CSV와 QC에만 기록되며, **평균에 들어가는 frame은 motion-correct하지 않는다.**

### 2. `mean_stack_frame_filtered`: single-stack의 현재 기본

위치: `bench2p_zstack_single.py:68-205`

- 한 plane 안의 usable frames로 per-pixel median reference를 만든다.
- 각 frame과 reference 사이 normalized Pearson correlation을 계산한다.
- `median(correlation) - 3 × 1.4826 × MAD`보다 낮은 frame을 제외한다.
- 최소 30%는 남긴다.
- 남은 frame을 **정렬 없이** arithmetic mean한다.
- 현재 2 µm run에서는 평균 0.18–0.67 frame/plane만 제외되었다. 이 수치는 artifact가 해결되었다는 증거가 아니라, adaptive correlation rule에서 outlier로 판정된 frame 수이다.

중요한 문서/코드 의미 차이:

- 일부 manifest 문구는 filter가 “large rigid shift” frame을 제거한다고 표현하지만, 실제 선택 기준은 rigid-shift estimate가 아니라 **raw intensity correlation outlier**이다.
- brightness fluctuation, noise, bleaching 또는 실제 구조 변화도 correlation에 영향을 줄 수 있다.

### 3. `merge_adjacent_stacks`: 두 stack 경계 정렬

위치: `bench2p_zstack.py:413-476`

- 첫 stack 마지막 plane과 두 번째 stack 첫 plane의 Z 위치가 0.25 µm 안에서 겹치는지 검사한다.
- 그 두 plane으로 XY rigid shift 하나를 구한다.
- 두 번째 stack 전체에 동일한 XY shift를 적용한다.
- 겹치는 두 plane을 `nanmean`하고 나머지를 이어 붙인다.

이것은 **stack 사이 XY offset correction**이지, 각 plane의 motion correction이나 Z drift correction이 아니다.

### 4. 현재 QC 판정

- plane별 early-vs-late XY shift magnitude의 p95가 3 px 이하이면 pass이다.
- max shift가 3 px를 넘어도 p95가 통과하면 전체 run은 pass일 수 있다. 실제 `scan9G1ZQTYA`가 p95 2.389 px, max 4.140 px로 pass이다.
- threshold가 px 단위이므로 3 px는 1 µm 자료에서 약 3.64 µm, 2 µm 자료에서 약 2.73 µm이다.
- early/late 5-frame window의 시간적 의미도 100 frames/plane과 60 frames/plane에서 다르다.
- sideview의 overall QC는 사실상 sampled positive int16 rail hit 여부로 정해지며 **입력 reconstruction의 `needs_review`를 자동 전파하지 않는다.**

## 정확히 무엇을 플롯하는가

### A. source reconstruction figure: `fig_bench2p_zstack_oblique`

현재 2 µm 대표 입력: `bench2p_zstack_single__20260901T173732Z__44d116a2`

입력 volume: 100×512×512 float32 mean OME-TIFF, spacing `(Z,Y,X) = (2.0, 0.91041015625, 0.91041015625) µm`

표시 변환:

```text
low  = 전체 volume intensity의 1 percentile
high = 전체 volume intensity의 99.8 percentile
u    = clip((F - low) / (high - low), 0, 1)
display = u^0.75
```

이 run에서 실제 `low=109.3051`, `high=1820.8167`이다. 이 변환은 figure용이고 저장된 float32 mean volume을 바꾸지 않는다.

| panel | 실제 계산 |
|---|---|
| a, axial | display-normalized volume을 Z축으로 maximum projection |
| b, side | Z를 `2.0 / 0.9104 = 2.1968×` 보간하여 대략 isotropic하게 만든 뒤 Y축 maximum projection |
| c, oblique | isotropic display volume을 azimuth −20°, elevation +28°로 회전한 뒤 maximum projection |

주의: figure JSON의 `plotted_data`는 `source_slice_metrics.csv`를 가리키지만, 이 CSV는 plane별 intensity 요약과 motion QC를 담을 뿐 **figure의 image pixel 값을 담지는 않는다.** 실제 image authority는 OME volume + code + display config 조합이다.

### B. 최신 2 µm cuboid: `fig_bench2p_zstack_cuboid`

run: `bench2p_zstack_sideview__20260902T084750Z__7f5a5ed3`

표시 변환:

```text
low  = 전체 volume의 50 percentile = 426.2833
high = 전체 volume의 99.99 percentile = 2954.4029
u    = clip((F - low) / (high - low), 0, 1)
display = asinh(2.5 × u) / asinh(2.5)
```

- source spacing으로 계산한 cuboid extent는 `Z 200.0 × Y 466.13 × X 466.13 µm`이다.
- 최대 dimension 360 px 제한 때문에 1.2948 µm isotropic grid로 resample한다.
- azimuth −35°, elevation +32° 회전 후, threshold 0.24와 opacity 0.14를 사용한 front-to-back alpha composite이다.
- cyan wireframe은 acquisition array의 12개 edge이며 해부학적 경계가 아니다.
- 이 패널은 MIP가 아니라 alpha composite이지만 여전히 display-only reconstruction이다.

여기서 200.0 µm는 `N voxels × spacing`으로 계산한 voxel-edge extent이고, 첫 plane
center에서 마지막 plane center까지의 ScanImage Z-coordinate 범위는 198.0 µm이다.
둘 중 무엇을 축의 “span”으로 부르는지 figure/table 전체에서 명시적으로 통일할 필요가 있다.

### C. `fig_bench2p_xz_thin_slabs`

- globally transformed display volume에서 Y축 6 µm 부근의 얇은 slab만 고른다.
- slab 안에서 Y 방향 maximum projection하여 X-Z 이미지를 만든다.
- 최신 run의 실제 세 slab은 Y center 116.53, 233.07, 348.69 µm이며 실제 두께는 각 6.373 µm이다.
- 전체 466 µm Y 깊이를 한꺼번에 접는 side MIP보다 overlap이 적지만, MIP와 axial PSF의 영향은 그대로 남는다.

### D. `fig_bench2p_candidate_side_reconstruction`

candidate 선택은 display volume이 아니라 원래 mean volume에서 수행한다.

```text
feature = Gaussian(volume - broad_Gaussian_background)
threshold = feature의 99.8 percentile
peak_local_max → 상위 6개 review target
```

- background sigma `(Z,Y,X) = (5,12,12) µm`
- feature sigma `(1.5,1.2,1.2) µm`
- 각 candidate를 중심으로 대략 half-size `(12,30,30) µm` ROI 선택
- 각 row: local XY MIP, 6 µm Y slab의 X-Z MIP, 6 µm X slab의 Y-Z MIP, local oblique attenuated MIP
- candidate는 cell segmentation이 아니며 feature peak일 뿐이다.

여기에도 물리 단위가 완전히 일관되지는 않는다.

- Gaussian sigma, ROI 크기, slab 두께는 µm 기반이다.
- `minimum_distance_voxels=10`과 `exclude_border_voxels=(8,24,24)`는 voxel 기반이다.
- 따라서 Z 방향 최소 거리/경계 제외가 1 µm와 2 µm stack에서 각각 10/8 µm와 20/16 µm가 되어 candidate selection이 동일하지 않다.

### E. raw saturation QC

- 최신 2 µm scan에서는 각 plane의 local frame `[0,15,30,45,59]`를 샘플링한다.
- 500 frames, 131,072,000 pixels에서 int16 음/양 rail hit를 센다.
- `scan9G1ZQTYA`의 sampled positive/negative rail fraction은 둘 다 0이었다.
- 이것은 샘플링된 frame이 int16 dtype rail에 닿지 않았다는 뜻일 뿐, 광학 포화나 detector/DAQ의 실제 유효 범위 전체를 배제하지는 않는다.

## 현재 코드가 Z-motion artifact에 답하지 못하는 지점

1. **Frame-by-frame XY registration 부재**
   low-correlation frame을 버릴 뿐, 남은 frame을 subpixel rigid/nonrigid 정렬하지 않는다.

2. **Cross-plane axial motion 추정 부재**
   plane을 순차적으로 이동하며 약 200–334초 동안 얻은 stack에서 동물이 Z 방향으로 움직였는지 추정하는 trajectory가 없다.

3. **Plane-to-plane deformation 구분 부재**
   실제 해부학적 Z 변화, axial PSF, stage step, tissue motion을 구분하는 model이나 reference volume이 없다.

4. **QC가 p95 한 값에 과도하게 축약됨**
   드문 큰 shift, 연속된 불량 plane, 특정 Z 구간의 변형을 놓칠 수 있다. 최소한 plane-index/Z/time에 따른 `dx,dy,error,corr` trace가 필요하다.

5. **현재 측정은 px 단위**
   다른 XY calibration 간 비교를 위해 µm 단위 shift와 acquisition-time-normalized rate를 함께 기록해야 한다.

6. **Z-step 비교의 acquisition confound**
   공통 물리 volume, 동일 animal/날짜, 동일 frames/plane, 동일 total duration 또는 명시적 model 조정 없이는 1 µm 대 2 µm 차이를 step 효과로 해석할 수 없다.

7. **MIP 기반 형태 판단의 한계**
   Z elongation을 더 눈에 띄게 하지만 원인이 motion인지 PSF인지 분리하지 않는다. 정량 형태 분석에는 PSF/bead calibration과 3-D segmentation 또는 model-based comparison이 필요하다.

8. **upstream QC 전파 부재**
   reconstruction이 `needs_review`여도 sideview가 saturation만 통과해 `pass`가 될 수 있다.

9. **figure plotted-data gap**
   현재 CSV는 geometry/peak/plane summary만 담고, normalized/resampled/rotated panel array 자체는 보존하지 않는다. `FIGURE_GUIDELINES.md`의 “exact plotted values” 기준에는 부족하다.

## 기존 자료로 가능한 개선 검증

새 촬영 전에도 같은 날 겹치는 stack을 이용해 일부 방법을 비교할 수 있다.

- **1 µm matched repeat:** ROS-2338 `scan9G1WRZL6`(pass)와 `scan9G1WRV2H`(큰 motion QC)의 거의 같은 0–100 µm 범위
- **2 µm matched/overlapping repeat:** ROS-2335 `scan9G1ZQQ9J`, `scan9G1ZQTYA`, `scan9G1ZQXUT`의 크게 겹치는 범위
- 한 stack으로 template을 만들고 다른 stack을 3-D register하여 repeatability를 측정할 수 있다.
- filter-only, per-frame 2-D rigid, per-frame nonrigid, cross-plane/3-D registration 후보를 같은 raw data에서 비교한다.
- 공통 physical grid와 공통 overlapping volume로 제한한 뒤 3-D normalized cross-correlation, plane-wise correlation, landmark displacement, candidate axial/lateral FWHM, repeat-scan localization error를 비교한다.
- synthetic known shifts와 motion-free phantom/bead stack을 추가하여 “보기 좋아짐”이 아니라 recovery error로 방법을 선택한다.

## 문헌 조사 에이전트에게 넘길 질문

아래 내용을 그대로 전달해도 된다.

> ScanImage bench2p slow Z-stack의 motion artifact 개선 문헌을 검토해 주세요. 데이터는 single-channel int16 TIFF, 512×512, 약 30 Hz이고, 각 Z plane에서 60 또는 100 frames를 연속 촬영한 뒤 다음 plane으로 이동합니다. Stack은 100 planes이며 Z-step은 1 또는 2 µm, 한 stack은 약 200–334초입니다. 현재 코드는 plane별 median-reference correlation low-outlier frame을 제외하고 mean하지만 frame 정렬은 하지 않으며, early-vs-late 5-frame XY phase-correlation은 QC로만 사용합니다. Cross-plane Z-motion correction은 없습니다. 다음을 primary paper와 공식 method/software 문서 중심으로 비교해 주세요: (1) within-plane rigid/nonrigid 2-D registration, (2) reference Z-stack을 이용한 axial displacement/Z-drift estimation, (3) sequential Z-stack의 3-D rigid/nonrigid registration 또는 distortion correction, (4) low-SNR two-photon frame에서 robust template/aggregation, (5) axial PSF와 motion-induced elongation을 구분하는 validation, (6) 1 µm와 2 µm sampling 비교 시 필요한 Nyquist/PSF/acquisition-time 고려. 각 방법에 대해 이 데이터 구조에 적용 가능성, 필요한 가정, failure mode, 계산량, 추천 QC metric, 검증 실험을 표로 제안해 주세요. 단순히 Suite2p/NoRMCorre 이름을 나열하지 말고 axial motion을 실제로 다루는지 분리해 주세요.

추가로 답해야 할 실험 설계 질문:

- 동일 plane 내 여러 frame만으로 true Z displacement를 식별할 수 있는가, 아니면 dense reference Z-stack/반복 volume이 필수인가?
- framewise 2-D registration 후 평균과 frame rejection 후 평균 중 low-SNR에서 어느 쪽이 bias가 적은가?
- Z-step 2 µm가 예상 axial PSF에 비해 충분한 sampling인지, 사용하는 objective/NA/wavelength/refractive index 정보를 어떻게 포함해야 하는가?
- 순차 촬영 중 breathing/pulsation/slow drift를 plane acquisition time에 맞춰 모델링한 사례가 있는가?
- correction 성능을 cell-like bright peak가 아니라 bead/landmark/repeated-volume consistency로 검증한 선례는 무엇인가?

## 권장 구현 우선순위

### P0: 해석 오류 방지

- figure와 manifest에 “axial motion correction 없음”을 계속 명시한다.
- sideview가 upstream `qc.json`의 `needs_review`를 전파하도록 한다.
- candidate `z_um`을 `z_relative_um`으로 바꾸고 `z_scanimage_um` 또는 plane별 Z-coordinate를 함께 보존한다.
- 1/2 µm 비교 figure에는 frames/plane, total duration, XY pixel size를 함께 표시한다.

### P1: metadata와 QC 강화

- `z_positions` monotonicity, `diff(z_positions)` regularity, `actualStackZStepSize` 일치 test/gate 추가
- drift를 px와 µm 모두 저장
- plane별 QC trace와 contiguous bad-plane count 추가
- `minimum_distance`와 border exclusion을 µm 기반 또는 isotropic-grid 기반으로 변경
- input QC, saturation QC, frame-filter QC를 분리해 overall status 계산

### P2: correction benchmark

- raw frame 보존 하에 per-plane framewise 2-D rigid correction + robust mean을 첫 baseline으로 구현
- 현행 filter-only 결과와 동일 stack에서 blind 비교
- 반복/겹침 stack을 reference로 cross-plane Z coherence와 3-D repeatability 평가
- 방법 선택 전 synthetic-shift 및 bead/phantom validation

### P3: figure traceability

- 각 image panel의 최종 float array 또는 lossless calibrated image를 machine-readable artifact로 저장
- CSV는 geometry/QC table로 정확히 명명하고, “plotted pixel data”라고 과장하지 않는다.
- 1 µm와 2 µm 모두 공통 physical crop/grid/display rule을 사용하는 비교 bundle을 별도로 만든다.

## 테스트와 reproducibility audit

현재 테스트가 확인하는 것:

- 반복된 Z position collapse
- synthetic XY rigid shift recovery
- 한-plane overlap merge
- projection 생성
- OME physical spacing read
- 2 µm Z spacing을 가진 cuboid resampling 예제
- µm-to-voxel slab 계산과 candidate physical coordinate 계산

빠져 있는 핵심 테스트:

- 실제/모의 ScanImage TIFF에서 1 µm와 2 µm `actualStackZStepSize` parsing
- parsed Z-step의 OME write/read round trip
- `actualStackZStepSize` 대 `diff(zs)` 불일치/비단조/누락 plane fail case
- `mean_stack_frame_filtered`의 known-shift/brightness-change/noise failure mode
- upstream QC propagation
- 1/2 µm에서 candidate detector의 physical-equivalence test

검토 시점의 실행 환경에는 `pytest` executable이 없어 test suite를 다시 실행하지 못했다. 위 평가는 코드, 기존 test, run manifest/config/QC/table과 생성 figure의 정적 검토에 근거한다.

추가 reproducibility 문제:

- 관련 Z-stack source 파일들은 현재 Git에서 untracked 상태이다.
- 2026-08-27 및 2026-09-01 reconstruction run에 기록된 helper hash는 `38b359...`이지만 현재 `bench2p_zstack.py` hash는 `27f11a...`이다.
- run은 과거 source code의 hash만 저장하고 source snapshot은 저장하지 않았다. 따라서 hash는 변경을 검출하지만, Git commit이나 run 내부 파일만으로 그 당시 helper를 복구할 수 없다.
- 최신 sideview run `...084750Z__7f5a5ed3`의 세 code hash는 현재 파일과 일치한다.
- `...20260901T211745Z__*` 네 sideview directory는 `analysis_manifest.json`이 없는 불완전 run이므로 완료 결과로 사용하면 안 된다.

## 최종 판단

현재 pipeline은 **metadata-aware physical visualization**에는 적절한 출발점이다. 1 µm와 2 µm Z-step을 읽고, OME에 기록하고, 후속 physical resampling에 사용하는 경로가 실제 산출물에서 확인된다.

반면 현재 결과를 **Z-motion-corrected volume** 또는 **정량적 3-D morphology**로 부르면 안 된다. 다음 단계의 핵심은 새로운 예쁜 projection보다, (1) plane 내부 frame registration, (2) 반복/겹침 stack을 이용한 axial/3-D consistency 평가, (3) PSF와 motion을 분리하는 validation, (4) acquisition confound를 통제한 1 대 2 µm 비교이다.
