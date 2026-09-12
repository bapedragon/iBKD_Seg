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

본 pilot은 L0/L1/L2별 독립 결과 archive를 회수한 뒤 `full_v1/` 아래에 함께
정리합니다. 한 shard만으로 loader를 선택하거나 중간 결과를 다음 shard 설정에
반영하지 않습니다.

Stage B 이후 결과도 완료된 v3 주 결과와 섞지 않고 이 계보 아래에 추가합니다.
