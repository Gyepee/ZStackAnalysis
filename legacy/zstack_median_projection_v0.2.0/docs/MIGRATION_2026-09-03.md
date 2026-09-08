# LabGraph Z-stack run migration

2026-09-03에 LabGraph의
`/home/jisooj/Workspace/LabGraph/datasets/analysis_runs/` 아래에서 이름이
`bench2p_zstack*`인 디렉터리를 이 프로젝트로 옮겼습니다.

- 완료 run: 30개, `analysis_runs/`, 합계 약 2.9 GB
- 미완료 run: 1개, `failed_runs/`
- 함께 이동된 figure 파일: 241개, 약 106.5 MB
- 원본 TIFF: 이동하거나 수정하지 않음
- LabGraph에 남은 해당 이름의 analysis run: 0개
- 분석 속도를 위해 만들었던 동일 파일명/크기의 local TIFF read copy 5개와 실행
  log는 `.staging/legacy_xyreg_read_copies_and_logs/`로 이동함(약 19 GB,
  authoritative raw가 아닌 삭제 가능한 cache)

기존 run 내부 manifest의 상대 경로는 생성 당시 LabGraph 위치를 가리키는 역사적
기록입니다. Immutable provenance를 훼손하지 않기 위해 기존 manifest는 수정하지
않았습니다. 새 run은 이 프로젝트 기준의 manifest를 새로 씁니다.

대규모 framewise XY-registration 초안은 최종 기능으로 채택하지 않고
`archive/xy_registration_wip/`에 분리했습니다. 새 기본 방법은 각 plane의 모든
frame을 pixelwise median으로 합치며, frame registration/rejection을 하지 않습니다.
