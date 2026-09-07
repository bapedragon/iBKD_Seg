# CUB-200-2011 실행 스크립트

데이터 감사, smoke, 분류 본실험, frozen probe와 결과 정리 스크립트를 이 폴더에
둡니다.

- `run_combined_smoke_b128.sh`: ResNet-56 teacher와 Vanilla, KD, LG, ALG-w20,
  iBKD-0.25, iBKD-0.5의 여섯 batch-128 분류 설정을
  각각 2 epoch 실행한 뒤, 여섯 frozen encoder × 세 LR probe를 각각 2 epoch
  실행하는 통합 smoke
- `run_full_shard_a_b128.sh`: Vanilla, LG, iBKD-0.5의 분류 3 seed와 frozen
  probe 전체를 실행하는 약 7시간 47분 shard
- `run_full_shard_b_b128.sh`: KD, ALG-w20, iBKD-0.25의 분류 3 seed와 frozen
  probe 전체를 실행하는 약 7시간 49분 shard

두 본실험 스크립트는 서로의 output을 입력으로 요구하지 않습니다. 각각 동일한
teacher와 데이터 archive를 재현하고, 결과 병합 시 teacher·초기화·split hash가
일치하는지 검사합니다.
