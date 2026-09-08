# ZStackAnalysis

`ZStackAnalysis`는 mini2p 장착 위치를 정하기 전, 수술 뒤 시간이 지나면서 후보
FOV와 깊은 plane의 가시성이 어떻게 회복되는지를 반복 Z-stack으로 관찰하는 독립
분석 프로젝트입니다. LabGraph가 데이터 접근·provenance·피겨 계약을 주관하고,
이 저장소는 Z-stack 재구성과 향후 depth-recovery 지표 개발을 담당합니다.

## 현재 기준선

- Pipeline ID: `zstack_median_projection`
- Version: `0.2.0`
- Stage: `initial_development`
- Analysis state: `exploratory`
- 재구성 단위: 물리적 Z-stack scan 1개 × ScanImage channel 1개
- 종단 생물학적 단위: animal
- 반복 측정: animal 안에 nested된 day/scan
- 기술적 관측: scan 안의 plane, plane 안의 frame; 생물학적 `n`으로 세지 않음

각 plane의 usable frame을 frame registration/rejection 없이 pixelwise median으로
합치고, float32 volume의 mean projection을 primary descriptive view로 만듭니다.
Maximum projection은 같은 표시 변환을 사용하는 비교 panel입니다. 이는 axial/Z
motion correction이 아니며, v0.2.0은 최적 장착일을 추정하지 않습니다.

## 필수 계약

모든 새 run은 다음을 포함합니다.

- `analysis_manifest.json`: 동물·날짜·scan·session·channel, 파이프라인 버전,
  질문·분석 단위·eligibility, 입력/코드/config/Git/output hash, supersession
- `config.snapshot.json`: 분석 명세·재구성 config·피겨 style·촬영 명세의 resolved snapshot
- 모든 표의 stable identity columns
- 각 피겨의 editable SVG, 300 dpi PNG, JSON panel map, Markdown note
- 피겨를 만든 exact float32 image arrays (`projection_source_arrays_float32.npz`)
- `README.md`, `environment.json`, `qc.json`, `logs/validation.json`

완료 run은 수정하지 않습니다. 변경된 결과는 새 `analysis_id`로 만들고
`--supersedes <old-analysis-id>`로 연결합니다. 기존 `analysis_runs/`는 legacy 기록으로
유지되며 새 계약을 소급해 덧씌우지 않습니다.

## 촬영 폴더의 JSON 명세

새 Z-stack 폴더에는 TIFF와 함께 정확히 `zstack_acquisition.json`을 둡니다.
[`config/zstack_acquisition.example.json`](config/zstack_acquisition.example.json)을
복사해 작성하며 스키마는
[`config/zstack_acquisition.schema.json`](config/zstack_acquisition.schema.json)입니다.

핵심 필드는 `data_kind: "zstack"`, animal/date/scan/session/source TIFF,
source channel, surgery date, post-surgery day, 촬영 목적, operator note, 알려진 QC
flag입니다. 파이프라인은 이 파일을 자동 발견하고 TIFF metadata와 대조합니다.
불일치하면 중단합니다. 기존 데이터에 이 파일이 없으면 재구성은 가능하지만
`legacy_metadata_incomplete` 및 longitudinal comparison `hold`로 기록됩니다.

## 실행

먼저 ResearchDataGovernance 또는 현행 LabGraph 데이터 접근 문서에 따라 실제
업로드 폴더와 TIFF를 확인한 뒤 실행합니다.

```bash
cd /home/jisooj/Workspace/ZStackAnalysis
PYTHONPATH=src python scripts/reconstruct_median_stack.py \
  --source-tiff /path/to/session/scan.tif
```

여러 channel이 저장됐고 촬영 JSON에 channel이 없다면 `--channel 3`처럼 명시합니다.
JSON 이름이 표준 이름이 아니면 `--acquisition-spec /path/to/spec.json`을 사용합니다.
로컬 read copy는 파일명·크기·SHA-256이 authoritative TIFF와 모두 같을 때만
`--read-copy`로 허용됩니다.

## 개발 순서

1. LabGraph/ResearchDataGovernance 계약으로 데이터를 발견하고 식별합니다.
2. LabGraph의 run·figure·immutability 운영 규칙을 확인합니다.
3. ZStackAnalysis 안에서 독립적으로 개발하고 테스트합니다.
4. 완성된 산출물을 manifest 기반으로 검증한 뒤 LabGraph가 등록·소비하게 합니다.

프로젝트 경계와 제안된 control-tower interface는
[`docs/GOVERNANCE.md`](docs/GOVERNANCE.md), LabGraph 문서 개선안은
[`docs/LABGRAPH_CONTROL_TOWER_RECOMMENDATIONS.md`](docs/LABGRAPH_CONTROL_TOWER_RECOMMENDATIONS.md)에
정리되어 있습니다.
