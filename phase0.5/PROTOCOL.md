# Phase 0.5 고정 프로토콜

상태: **v1 고정(LOCKED) — 결과 확인 전 확정**

고정일: 2026-09-04

machine-readable 설정:
[`configs/flowers102_phase05_v1.json`](configs/flowers102_phase05_v1.json)

> 이 config는 실제 실행 당시의 SHA-256을 보존하기 위해 내부 `protocol_id`의 과거
> `phase1a` 문자열을 바꾸지 않았습니다. 현재 단계명과 저장 위치는 Phase 0.5입니다.

## 목적과 해석 한계

분류용 encoder를 완전히 고정한 뒤 마지막 patch-grid feature에서 전경/배경을
선형적으로 얼마나 잘 읽을 수 있는지 검사합니다. 모든 방법에 동일한
`Conv2d(192, 2, 1)` probe를 적용하므로, 측정 대상은 강한 decoder 성능이 아니라
최종 feature에 남은 공간정보의 선형 복원성입니다.

Flowers-102 segmentation은 사람이 만든 정답이 아니라 자동 생성된
pseudo-mask입니다. 따라서 데이터·feature cache·probe·metric·시각화 경로의
진단에만 사용하며 방법 간 순위를 논문의 과학적 결론으로 사용하지 않습니다.
주장은 Oxford-IIIT Pet의 공식 trimap을 사용하는 Phase 1에서 검증합니다.

## 프로토콜 선택 원칙

- 결과를 본 뒤 유리한 transform, seed, learning rate, 제외 규칙을 고르지 않습니다.
- encoder는 `eval()` 상태로 완전히 고정하고 선형 head만 학습합니다.
- 모든 방법에 동일한 사전 고정 learning-rate grid를 적용하고 validation에서만
  선택합니다.
- test는 learning rate와 epoch 선택이 끝난 뒤 seed별로 한 번만 평가합니다.
- 입력·마스크의 기하 변환과 metric 집계는 모든 방법에 동일하게 적용합니다.

참고한 공개 계약은 DINO/DINOv2의 frozen linear evaluation 구현과 Oxford 공식
데이터 계약입니다.

- https://github.com/facebookresearch/dino/blob/main/eval_linear.py
- https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py
- https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

## 데이터 계약

- 데이터: Oxford Flowers-102 공식 이미지와 공식 blue-screen segmentation
- split: 공식 train 1,020 / validation 1,020 / test 6,149
- mask 변환: 원본과 composite에서 alpha를 복원한 뒤 `alpha >= 0.5`를 전경으로 정의
- 제외: 없음. 빈 mask와 거의 전경뿐인 mask도 모두 포함
- 사후 정제: morphology, connected component, 수작업 수정 모두 금지
- 입력: RGB를 `224 x 224`로 직접 bilinear resize하고 ImageNet mean/std로 정규화
- mask: 같은 좌표계의 `224 x 224` binary mask를 nearest-neighbor로 생성
- random augmentation: 없음

## Encoder와 feature 계약

- architecture: `deit_tiny_patch16_224`
- checkpoint: `manifests/checkpoints.json`의 byte size와 SHA-256이 일치해야 함
- strict state-dict loading 필수
- encoder mode: `eval()`; encoder gradient 0개
- feature: 마지막 block(index 11), `norm=False`, NCHW
- shape: `B x 192 x 14 x 14`; cache dtype float32

`norm=False`는 Phase 0에서 감사한 feature 계약입니다. post-norm이나 다른 layer는
결과를 본 뒤 바꾸지 않고 Phase 2 민감도 분석으로 분리합니다.

## Probe 학습 계약

- head: bias 포함 `Conv2d(192, 2, kernel_size=1)` — 386 parameters
- 초기화: weight `Normal(0, 0.01)`, bias 0
- target: `224 x 224` mask를 area downsample한 patch occupancy가 0.5 이상이면 전경
- loss: unweighted 2-class cross entropy
- optimizer: SGD, momentum 0.9, weight decay 0, Nesterov 사용 안 함
- learning-rate grid: `[0.01, 0.03, 0.1]`
- schedule: 100 epochs cosine decay, minimum learning rate 0
- batch size: 64; probe seeds: `[1, 2, 3, 4, 5]`
- augmentation과 stochastic encoder 연산은 사용하지 않음

## 선택과 평가 계약

각 checkpoint와 probe seed에 대해 세 learning rate를 모두 train split으로
학습합니다. 각 run에서 validation `14 x 14` grid mIoU가 가장 높은 epoch를
보존하고, 그중 validation mIoU가 가장 높은 learning rate 하나를 선택합니다.
동률이면 낮은 learning rate, 그다음 이른 epoch를 선택합니다.

선택된 probe만 test에 적용합니다.

- 주 metric: `224 x 224`로 bilinear upsample한 예측의 2-class mIoU
- 보조 metric: foreground IoU, background IoU, foreground Dice, pixel accuracy
- 진단 metric: `14 x 14` grid의 동일 metric
- 집계: 이미지별 평균이 아닌 split 전체 global confusion matrix
- probe seed 요약: 평균, 표준편차와 각 seed 원값
- baseline: all-background와 train mean-mask
- 정성 표본: test ID `[1, 45, 568, 1737, 3650, 5775, 8189]`

정성 ID는 결과를 보기 전에 고정했으며, 빈 pseudo-mask ID 45와 거의 전경뿐인
pseudo-mask ID 568을 의도적으로 포함했습니다. panel 순서는 입력, pseudo-mask,
예측입니다.

## 완료 gate

1. 데이터와 checkpoint hash 검증 통과
2. cache의 feature shape·dtype·ID 순서 검증 통과
3. encoder gradient 0, probe gradient 2 tensors 확인
4. train loss와 validation metric이 finite
5. validation 선택 뒤 test metric과 정성 mask 생성
6. Ours/ALG는 조건 일치 주 진단군, KD는 조건 불일치 탐색군으로 분리 보고

이 gate는 모두 통과했습니다. Pet Phase 1의 trimap 경계 ignore 규칙, split,
teacher, encoder seed와 방법별 설정은 본 실험 결과를 보기 전에 별도 v1 config로
고정합니다.
