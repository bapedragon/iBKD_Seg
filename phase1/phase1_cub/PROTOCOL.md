# CUB-200-2011 Phase 1 프로토콜

이 문서는 완료된 ResNet-50/224 v3 계열 본학습의 공통 계약을 요약합니다. 정확한
machine-readable 값은
[`configs/cub200_r50_224_b128_full_v3.json`](configs/cub200_r50_224_b128_full_v3.json)을
기준으로 합니다.

## 데이터와 분할

- CUB-200-2011: 200종, 총 11,788장
- 공식 train 5,994장에서 class별 3장을 seed 2027 규칙으로 validation에 고정
- 최종 train/validation/official test: `5,394 / 600 / 5,794`
- validation image ID SHA-256:
  `263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854`
- 분류 encoder 학습에는 RGB와 class label만 사용하며 mask, box, part와 attribute는
  사용하지 않음

## Teacher

- TorchVision ResNet-50, 224×224, scratch(`weights=None`), seed 1
- batch 128, 200 epoch, SGD lr 0.05, momentum 0.9, weight decay 1e-4
- 5-epoch linear warm-up 뒤 cosine decay
- validation macro Top-1 최대 checkpoint를 선택하고 동률이면 이른 epoch
- 모든 guided student가 감사된 동일 checkpoint 하나를 공유

Teacher 결과와 hash는
[결과 보고서](reports/classification/resnet50_224_teacher_v3/RESULTS.md)에 있습니다.

## Student 분류

- DeiT-Tiny/16, 224×224, scratch, encoder seeds `[1,2,3]`
- 300 epoch, AdamW lr 5e-4, weight decay 0.05, 20-epoch LR warm-up 뒤 cosine
- seed 안에서 모든 방법의 초기 state와 batch order를 맞춤
- LG: student blocks `[0,6,11]` ↔ teacher stages `[layer2,layer3,layer4]`
- ALG-w20: LG 기반, controller warm-up 20
- iBKD: 모든 12개 student block, λ=`0.25`와 `0.5`를 모두 보고
- 분류 checkpoint 선택: validation macro Top-1 최대, 동률이면 이른 epoch
- official test는 선택 완료 뒤 checkpoint당 1회이며 방법·λ 선택에 사용하지 않음

현재 감사가 끝난 v3 본결과는 guided 네 방법입니다. Vanilla/KD가 없는 상태를
6방법 전체 결과라고 표기하지 않습니다.

## Frozen segmentation probe

- validation-selected 분류 encoder를 strict load하고 `eval()`·완전 freeze
- 마지막 block의 pre-norm patch feature `[192,14,14]`
- 공통 head `Conv2d(192,2,1,bias=True)`, 총 386 parameters
- probe seeds `[1,2,3,4,5]`, LR `[0.01,0.03,0.1]`, 100 epoch
- validation grid mIoU로 LR/epoch 선택
- 모든 선택이 끝난 뒤 official test input-224 2-class mIoU를 probe당 1회 평가

3-seed guided 결과는
[frozen probe 보고서](reports/frozen_probe/resnet50_224_b128_guided_3seed_v5/RESULTS.md)를
기준으로 합니다.

## 직접 공간정보 진단

- 주 지표: frozen part-localization probe의 PCK@0.1
- 보조 지표: teacher layer3 ↔ student block별 spatial CKA
- 보조 지표: attention rollout의 patch AP, pointing, foreground mass
- Part probe만 validation으로 선택하며 CKA에는 validation 600장을 사용
- Attention은 고정 규칙으로 official test를 1회 평가

정확한 정의와 결과는 [DIRECT_SPATIAL_PROTOCOL.md](DIRECT_SPATIAL_PROTOCOL.md) 및
[3-seed 결과](reports/direct_spatial/resnet50_224_b128_guided_3seed_v2/RESULTS.md)에
있습니다.

## 결과 보존

- Git: protocol config, 정리된 본학습 JSON/CSV/PNG, 결과 해석, checkpoint hash
- GitHub Release: 큰 teacher/student/probe checkpoint와 원시 실행 증거
- Git 제외: CUB 데이터 archive, feature/target cache, 중복 원시 출력
- 완료된 본실험의 선행 실행 점검 산출물은 보존하지 않음

## 사후 재현성 gate

main-L0 iBKD λ=0.25 seed 1의 반복 실행에서 분류 정확도와 guidance controller
종료 시점이 크게 달랐습니다. 따라서 레이어 연결 원인 분석을 계속하기 전에
[제어 A/A 재현성 프로토콜](reproducibility/PROTOCOL.md)의 2-epoch smoke와
300-epoch full A/A를 순서대로 수행해야 합니다. iBKD의 CUDA deformable-conv
backward는 결정론 구현이 없어 완전 bitwise 결정론을 주장하지 않으며, 동일한
입력·RNG 아래 장기 결과 변동을 직접 측정합니다. 이 진단은 기존 v3 주 결과를
덮어쓰지 않으며 official test나 방법 선택에 사용하지 않습니다.
