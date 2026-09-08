# ZStackAnalysis

Z-stack 전용 분석 부서의 재구축 작업 공간입니다.

현재는 새 파이프라인을 설계하기 전에 기존 방법의 피겨를 검토하는 단계입니다.
동결한 코드와 기존·신규 분석 결과는
[`legacy/zstack_median_projection_v0.2.0`](legacy/zstack_median_projection_v0.2.0)에
함께 보관합니다.

새 파이프라인의 최소 운영 순서는 다음과 같습니다.

1. LabGraph와 DataGovernance의 데이터 접근 계약을 확인한다.
2. `collection_manifest.json`과 raw TIFF를 대조해 분석 가능성을 판정한다.
3. 승인된 Z-stack을 분석한다.
4. 결과가 완성되면 Discord webhook으로 실행 identity와 결과 위치를 알린다.

Discord 연동과 새 구현은 legacy 피겨 검토 뒤 시작합니다.
