# CUB 이미지 loader 탐색 결과

완료된 CUB ResNet-50/224 v3 결과를 덮어쓰지 않는 사후 탐색 실험을 보존합니다.

- `damage_audit_v1/`: 학습 전 crop이 part, foreground mask, bounding box를 얼마나
  보존하는지 측정한 결과. [쉬운 해석과 수치표](damage_audit_v1/RESULTS.md),
  machine-readable JSON·CSV, 입력 로그 출처 manifest를 함께 보존합니다.
- `failed_smoke_v1_issue746/`: 분류 12개 뒤 첫 frozen segmentation probe 생성
  전에 schema 누락으로 중단된 Stage B v1의 실패 기록입니다. Metric이나
  official-test 결과로 사용하지 않습니다.
- `smoke_v2/`: 수정된 Stage B smoke의 완료 gate, 시간·메모리와 12개 진단값을
  보존합니다. 비과학적 2-epoch 값으로 loader를 선택하지 않습니다.
- `smoke_l1_l2_v2/`: L0 실제시간 확인 뒤 L1·L2를 한 H200 이슈로 묶기 위해 수행한
  issue 750 운영 smoke의 gate와 시간·메모리를 보존합니다. 성능값은 선택에 쓰지
  않습니다.
- `full_v1_log_snapshot/`: L0와 L1·L2 완료 로그에서 추출한 12개 방법별 수치와
  profile 평균, 해석을 보존합니다. L2가 사전 선택 규칙에서 1위지만 결과 archive와
  checkpoint 감사 전까지는 잠정 로그 스냅샷입니다.

L0/L1/L2 완료 로그는 모두 도착해 `full_v1_log_snapshot/`에 함께 정리했습니다.
다만 독립 결과 archive는 아직 회수하지 않았으므로 checkpoint와 원본
machine-readable 결과를 감사한 뒤 `full_v1/` 최종 보고서를 만듭니다. 로그
스냅샷은 이 상태를 명시하며 archive 기반 결과로 가장하지 않습니다.

Stage B 이후 결과도 완료된 v3 주 결과와 섞지 않고 이 계보 아래에 추가합니다.
