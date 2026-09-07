# Phase 1 CUB-200-2011 프로토콜

상태: **초안 체크리스트 — LOCK 전**

이 문서는 결과나 본실험 코드를 만들기 전에 CUB-200-2011 전용 실험 계약을
확정하기 위한 자리입니다. 아래 항목이 모두 결정되고 machine-readable config의
해시가 기록되기 전에는 H200 본실험을 시작하지 않습니다.

## 고정할 항목

- 공식 이미지·분류 annotation·segmentation mask의 출처, byte size, SHA-256
- 이미지 ID와 class label, 공식 split, mask의 1:1 대응 및 누락 처리
- train/validation/test 분할과 test-once 정책
- pixel class 정의, resize/interpolation, 경계 또는 무효 픽셀 처리
- CNN teacher와 ViT student 구조 및 초기화
- Vanilla, KD, LG, ALG, iBKD의 공통 설정과 방법별 고정값
- batch size, epoch, optimizer, learning-rate schedule과 seed
- 분류 checkpoint의 validation-only 선택 기준
- frozen feature 위치·shape·normalization과 encoder freeze 검사
- 공통 probe 구조, LR 후보, epoch, seed와 validation 선택 기준
- IoU, mIoU, Dice 등 최종 metric과 비영상 baseline
- smoke와 본실험의 분리, 산출물 manifest와 결과 보고 형식

## Pet과의 관계

Pet은 완료된 독립 실험이므로 CUB 결과에 맞춰 Pet의 LOCK config나 결과를
수정하지 않습니다. 가능한 조건은 Pet과 맞추되, CUB 데이터 특성 때문에 달라지는
항목은 근거와 함께 이 문서에 명시합니다.

## 본실험 전 통합 smoke 계약

본실험 결과를 보기 전 실행 경로를 검증하기 위한 batch-128 통합 smoke만
`configs/cub200_b128_combined_smoke_v1.json`에 별도로 고정했습니다. 이는 이
문서의 본실험 LOCK을 대신하지 않습니다.

- 공식 train 5,994장만 클래스별 고정 분할: train 5,394 / validation 600
  (클래스당 3장, seed 2027)
- 공식 test 5,794장: loader 생성·이미지/mask decode·평가 모두 금지
- scratch CIFAR-style ResNet-56 teacher 1개, 입력 32, batch 128, seed 1,
  2 epoch
- scratch DeiT-Tiny student 6개: Vanilla, KD, LG, ALG, iBKD λ=0.25/0.5,
  입력 224, batch 128, seed 1, 각각 2 epoch
- canonical ALG controller warm-up 0, iBKD controller warm-up 20
- 각 student 최신 smoke checkpoint를 strict load하고 encoder를 완전히 동결
- 공식 binary mask의 grayscale 값 `> 0`을 foreground로 임시 고정하고 nearest
  resize
- 1×1 probe, seed 1, LR `[0.01, 0.03, 0.1]`, 후보별 2 epoch, validation만 평가
- 모든 smoke 정확도·IoU는 비과학적 진단값이며 본실험 선택과 논문 주장에 사용 금지

Smoke가 통과해도 archive SHA-256, 본실험 checkpoint 선택·test-once 계약,
classification/probe seed, epoch와 정성 샘플을 확정한 새 full config가 필요합니다.

현재 확정한 반복 원칙은 smoke에서는 encoder seed 1만 사용하고, 본실험 분류에서는
encoder seed `[1, 2, 3]`을 모두 실행한다는 것입니다. Probe 반복 수 등 나머지
본실험 항목은 full config LOCK 때 확정합니다.
