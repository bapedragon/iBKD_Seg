# CUB-200-2011 실행 스크립트

데이터 감사, smoke, 분류 본실험, frozen probe와 결과 정리 스크립트를 이 폴더에
둡니다.

- `run_r50_224_guided_smoke_b128.sh`: 현재 v3용. Scratch ResNet-50/224 teacher와
  LG, ALG-w20, iBKD-0.25, iBKD-0.5를 각각 2 epoch 실행하고 네 frozen
  encoder × 세 LR probe를 2 epoch 실행합니다. Classification과 선택된 probe의
  official test 경로, peak memory, 4방법×3seed guided block 시간 외삽까지 기록합니다.

아래 두 스크립트는 미실행 ResNet-56/32 v2 보존본입니다.

- `run_combined_smoke_b128.sh`: ResNet-56 teacher와 Vanilla, KD, LG, ALG-w20,
  iBKD-0.25, iBKD-0.5의 여섯 batch-128 분류 설정을
  각각 2 epoch 실행한 뒤, 여섯 frozen encoder × 세 LR probe를 각각 2 epoch
  실행하는 통합 smoke
- `run_full_guided_b128.sh`: teacher를 한 번 학습하고 ALG-w20, iBKD-0.25,
  iBKD-0.5의 분류 3 seed와 frozen probe 전체를 실행하는 약 8시간 작업

현재 v3 본실험 스크립트는 v3 smoke 결과로 MIG 용량과 10시간 분할을 정한 뒤
추가합니다.
