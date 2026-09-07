# CUB-200-2011 실행 스크립트

데이터 감사, smoke, 분류 본실험, frozen probe와 결과 정리 스크립트를 이 폴더에
둡니다.

- `run_combined_smoke_b128.sh`: ResNet-56 teacher와 Vanilla, KD, LG, ALG-w20,
  iBKD-0.25, iBKD-0.5의 여섯 batch-128 분류 설정을
  각각 2 epoch 실행한 뒤, 여섯 frozen encoder × 세 LR probe를 각각 2 epoch
  실행하는 통합 smoke
- `run_full_guided_b128.sh`: teacher를 한 번 학습하고 ALG-w20, iBKD-0.25,
  iBKD-0.5의 분류 3 seed와 frozen probe 전체를 실행하는 약 8시간 작업

Vanilla, KD, LG용 두 번째 스크립트는 첫 결과의 teacher checkpoint를 파일 및
model-state SHA-256으로 고정한 뒤 추가합니다. 두 번째 작업은 teacher를 재학습하지
않습니다.
