# Phase 1 Oxford-IIIT Pet 분류 프로토콜과 full-run 계약

상태: **본 분류 실험 LOCK — batch 64/128과 iBKD λ 0.25/0.5 모두 보고**

고정일: 2026-09-05

machine-readable 설정:
[`configs/oxford_iiit_pet_phase1_v1.json`](configs/oxford_iiit_pet_phase1_v1.json)

config SHA-256:
`38f743958d1211144495dd9b4c7eb6edd4c12ab1bacbb27c75d38528b3e72143`

## 목적

품종 분류 정답만으로 학습한 iBKD encoder의 마지막 `14 x 14` patch feature에
동물의 위치와 형태 정보가 Vanilla, KD, LG, ALG보다 선형적으로 더 쉽게 읽히는
상태로 남아 있는지 검증합니다.

Phase 1에는 연결된 두 평가가 있습니다.

1. Pet 37개 품종 분류 성능을 측정합니다.
2. 분류 validation으로 선택한 동일 checkpoint의 encoder를 고정하고 공식
   trimap에 공통 spatial probe를 학습하여 IoU와 Dice를 측정합니다.

분류 결과는 독립적인 사전 관문이 아닙니다. iBKD가 분류 1위를 해야 probe를
실행하는 것도 아닙니다. 분류 정확도는 각 encoder의 원래 능력을 기록하고 probe
차이가 단순한 분류 성능 차이로 설명되는지 판단하는 동반 지표입니다.

## 선택 원칙과 근거

- Oxford-IIIT Pet 원 논문은 37개 품종, 품종 label, head ROI, foreground/background/
  ambiguous trimap과 평균 class accuracy 평가를 정의합니다.
  <https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf>
- 공식 배포 파일은 `trainval`과 `test`를 제공하며 torchvision도 같은 두 split과
  category/segmentation target을 지원합니다.
  <https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html>
- student 기본값은 AAAI 제출 당시 동결한 Ours V1/iBKD와 DeiT/LG 계열의
  data-scarce scratch 설정을 따릅니다.
  <https://github.com/facebookresearch/deit>
  <https://arxiv.org/abs/2207.10026>
- probe는 강한 decoder의 성능을 겨루지 않고 frozen feature의 선형 복원성을
  측정합니다. 따라서 Phase 0.5에서 검증한 1x1 probe 계약을 그대로 유지합니다.

여기서 “보편적인 방식”은 Pet 최고 정확도를 만들기 위한 ImageNet fine-tuning이
아니라, 비교 대상 하나만 바꾸고 나머지 조건을 통제하는 KD 실험 방식을 뜻합니다.
외부 pretraining은 이미 학습된 공간정보를 모든 encoder에 주입하여 iBKD가
분류-only 학습 중 보존한 정보를 보기 어렵게 만들 수 있으므로 v1에서 사용하지
않습니다. pretrained 비교가 필요하면 결과 확인 후 별도 protocol family로
추가합니다.

iBKD에는 서로 다른 두 역사적 설정이 있습니다. `AAAI_ours_submission`과 Phase
0.5 matched checkpoint 계보는 `lambda=0.5`, batch `64`이고, 이후 공개
`IBKD_AAAI-27` 기본값은 `lambda=0.25`, batch `128`입니다. 12-way timing에서 모든
조합이 H200 한 장에 들어가고 두 batch의 예상시간도 비슷함을 확인했습니다. 따라서
성능을 보기 전에 batch `64`와 `128`을 독립 profile로 고정하고, 각 profile에서
Vanilla/KD/LG/ALG/iBKD λ=0.25/iBKD λ=0.5를 모두 3 seed 실행합니다. 결과가 좋은
batch나 λ 하나만 사후 선택하지 않고 두 profile과 두 λ를 모두 별도로 보고합니다.

## 데이터 계약

### 공식 파일

| 파일 | byte | MD5 | SHA-256 |
|---|---:|---|---|
| `images.tar.gz` | 791,918,971 | `5c4f3ee8e5d25df40f4fd59a7f44e54c` | 전체 다운로드 감사 전 |
| `annotations.tar.gz` | 19,173,078 | `95a8c909bbe2e81eed6a22bccdf3f68f` | `52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91` |

공식 URL은 다음 두 개만 허용합니다.

- <https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz>
- <https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz>

확인한 annotation 내부 파일 SHA-256은 다음과 같습니다.

| 내부 파일 | SHA-256 |
|---|---|
| `annotations/list.txt` | `6a54ab256e22f7a33c6f17a7669e58ea5f6f9c7a080ec2622c205aefd4b354da` |
| `annotations/trainval.txt` | `408f3f609481b939c94634169e6413414b733a3faeba440cbdcc5c02142eebdc` |
| `annotations/test.txt` | `a5454003774ffe01f4f322756d3ba5495bae21cb30bb217ab285dbfa2bef245c` |

공식 archive에는 split에 포함되지 않은 추가 파일도 있으므로, 표본 universe는
오직 `trainval.txt` 또는 `test.txt`에 등장하는 7,349개 ID로 정의합니다. 각 ID는
고유해야 하고 image, 품종 label, trimap이 1:1로 존재해야 합니다. decode 실패나
trimap 값 `{1,2,3}` 이외의 값이 하나라도 있으면 full run을 시작하지 않습니다.

### 고정 split

```text
공식 trainval 3,680
  ├─ classification/probe train 2,940
  └─ classification/probe validation 740 = 품종별 20장

공식 test 3,669 — 최종 평가 전용
```

- validation은 공식 `trainval` 안에서 품종별 정확히 20장을 선택합니다.
- split seed는 `2027`입니다.
- 각 품종에서 `sha256("2027:{image_id}")`, 그다음 `image_id` 순으로 정렬한 앞
  20장을 validation으로 사용하고 나머지를 train으로 사용합니다.
- 생성된 ID 목록과 SHA-256을 manifest로 보존합니다.
- teacher, 모든 student 설정, 모든 probe가 같은 `2,940/740/3,669` partition을
  공유합니다.
- official test는 optimizer step, epoch 선택, LR 선택, controller 선택에 사용하지
  않습니다.

품종별 20장으로 고정한 stratified validation은 공식 test를 매 epoch
확인하는 누수를 막고, 모든 품종이 validation에 동일하게 기여하도록
하기 위한 선택입니다. 전체 비율은 `2,940/740`으로 약 80/20이지만,
원래 100장보다 적은 품종이 있어 각 품종 내 비율을 정확히 80/20으로
표현하지는 않습니다.

## 공통 CNN teacher

| 항목 | 고정값 |
|---|---|
| 모델 | CIFAR-style ResNet56 (`6n+2`, `n=9`) |
| 초기화 | scratch |
| 입력 | `32 x 32` |
| 분류 class | 37 |
| seed | `1` |
| epoch | `300` |
| train/eval batch | `128/200` |
| optimizer | SGD, LR `0.1`, momentum `0.9`, Nesterov |
| weight decay | `5e-4`; bias와 normalization parameter 제외 |
| schedule | warm-up 없이 cosine decay to `0` |
| label smoothing | `0` |
| Mixup/CutMix | 사용 안 함 |
| precision | FP32 |
| checkpoint 선택 | validation macro Top-1 최대, 동률이면 이른 epoch |
| test | 선택 후 한 번 |

선택된 teacher checkpoint 하나를 hash로 고정하고 KD/LG/ALG/iBKD의 모든 seed가
공유합니다. 방법마다 별도 teacher를 고르지 않습니다. Vanilla는 teacher를
사용하지 않습니다.

## 공통 DeiT-Tiny 분류 계약

### 모델과 반복

| 항목 | 고정값 |
|---|---|
| student | `deit_tiny_patch16_224` |
| 초기화 | scratch, 외부 pretrained weight 없음 |
| 입력 | `224 x 224` |
| encoder seed | `[1, 2, 3]` |
| epoch | `300` |
| train batch profile | `[64, 128]` — 두 profile 모두 본 실험 및 별도 보고 |
| eval batch | `200` |
| precision | FP32 |

각 seed별 최초 student state를 먼저 한 번 생성하여 hash로 저장합니다. 같은 seed의
Vanilla/KD/LG/ALG/iBKD는 classifier를 포함해 byte-identical한 student state에서
시작해야 합니다. method module 초기화가 student 초기화를 바꾸지 않도록 RNG를
분리합니다. 같은 seed는 같은 split 순서와 augmentation worker seed를 사용합니다.

### 최적화와 regularization

| 항목 | 고정값 |
|---|---|
| optimizer | AdamW, betas `(0.9, 0.999)`, eps `1e-8` |
| LR | `5e-4` |
| minimum LR | `5e-6` |
| weight decay | `0.05` |
| no-decay | 없음; 제출본처럼 모든 학습 parameter에 동일 적용 |
| LR schedule | 20 epoch linear warm-up 후 cosine |
| warm-up 시작 | `5e-7` = LR × `0.001` |
| label smoothing | `0` |
| drop path | `0.1` |
| dropout | `0` |
| Mixup/CutMix | 사용 안 함 |
| model EMA | 사용 안 함 |
| gradient clipping | 사용 안 함 |
| train `drop_last` | 사용 |

본 과학 비교에서는 optimizer parameter grouping과 **train batch까지** 모든 방법에
동일하게 적용합니다. 필요한 adapter와
iBKD module을 포함한 모든 학습 parameter에 같은 weight decay를 적용합니다. Pet 결과를 보고 방법별
LR, batch, augmentation, epoch를 따로 조정하지 않습니다.

### 입력 변환

분류 train:

- random resized crop `224`, scale `[0.08, 1.0]`, ratio `[3/4, 4/3]`, bicubic
- horizontal flip probability `0.5`
- RandAugment `rand-m9-mstd0.5-inc1`
- random erasing `p=0.25`, pixel mode, 1 region
- ImageNet mean/std normalization

`color_jitter=0.4` 인자는 공식 LG/DeiT 계열 호출과 같이 기록하지만 RandAugment
경로에서 별도 ColorJitter가 실제 transform에 중복 적용되지 않도록 realized
transform 목록을 시작 로그에 출력합니다.

Teacher는 독립 random crop을 받지 않습니다. student와 동일한 augmented RGB view를
bilinear로 `32 x 32`로 내린 뒤 teacher normalization을 적용합니다. 이로써 두
branch의 crop/flip 위치가 달라지는 문제를 막습니다.

분류 validation/test:

- 전체 이미지를 `224 x 224`로 직접 bilinear resize
- center crop과 test-time augmentation 없음
- ImageNet mean/std normalization

직접 resize는 LG/iBKD 실험 계열과 맞고 이후 trimap probe와 정확히 같은 좌표계를
제공합니다.

## 방법별 고정 계약

공통 설정에서 달라질 수 있는 것은 teacher 사용 여부, distillation objective,
필요한 feature adapter와 문서화된 controller뿐입니다.

### Vanilla

```text
L = CE
```

Teacher와 distillation module을 생성하지 않습니다.

### KD

```text
L = (1 - alpha) CE + alpha T^2 KL(student/T, teacher/T)
T = 4.0, alpha = 0.9
```

최종 class logits만 전달하고 spatial feature adapter는 사용하지 않습니다.

### LG

- student block `[0, 6, 11]`, teacher stage `[0, 1, 2]`
- stage별 학습 가능한 `1 x 1` channel projection
- 두 feature를 더 큰 grid로 bilinear 정렬
- stage mean MSE를 합산
- 전체 300 epoch 동안 `L = CE + 2.5 L_LG`

### ALG

LG feature objective에 공개 ALG 식을 적용합니다.

- `beta_on=2.5`, threshold `-0.02`
- loss smoothing window `50`, derivative smoothing window `50`
- controller 전용 warm-up 없음; optimizer LR warm-up 20 epoch와 구분
- smoothed derivative가 threshold 이상이면 guidance를 영구 종료

### iBKD

구조와 controller는 AAAI 제출본 Ours V1/iBKD를 사용하고, λ 두 값을 사전
지정된 비교 설정으로 모두 실행합니다.

```text
L_feature = lambda L_fuse + (1 - lambda) L_align
lambda = {0.25, 0.5}; 둘 다 실행·보고
L = CE + beta(e) L_feature
```

- DeiT-Tiny 12개 block 전체의 학습 가능한 stage별 convex aggregation
- stage별 student/teacher feature 중 더 큰 grid로 bilinear 정렬(`32/16/14`)
- stage별 `1 x 1` projection
- `5 x 5` deformable spatial attention
- 4-head convolutional cross-attention, `1 x 1` Q/K/V
- `beta_on=2.5`, threshold `-0.02`, 두 smoothing window `50`
- controller stop warm-up `20` epoch
- smoothed derivative가 threshold보다 커지면 guidance를 영구 종료

ALG와 iBKD controller의 경계 및 warm-up 차이는 각 공개/현재 방법 정의의 일부로
고정합니다. 이 외의 학습 조건은 같게 유지합니다. controller 자체까지 동일하게
만드는 인과 대조가 필요하면 Phase 2의 별도 paired ablation으로 수행합니다.

## 분류 checkpoint 선택과 보고

각 방법·encoder seed는 매 epoch validation만 평가합니다.

1. validation macro Top-1이 가장 높은 checkpoint를 선택합니다.
2. 동률이면 더 이른 epoch를 선택합니다.
3. 선택된 checkpoint를 strict load합니다.
4. official test를 정확히 한 번 평가합니다.
5. 바로 그 checkpoint를 segmentation probe에 전달합니다.

분류 표는 다음을 보고합니다.

- 주 지표: test macro Top-1(품종별 정확도의 평균)
- 보조 지표: overall Top-1, Top-5
- seed `[1,2,3]` 원값과 mean ± sample SD
- 같은 seed끼리의 `iBKD - ALG` paired difference

Segmentation 결과를 보고 다른 분류 epoch를 고르는 것은 금지합니다.
encoder seed가 3개뿐이므로 `p > 0.05`를 근거로 “통계적으로 동등하다”고 쓰지
않습니다. 별도의 equivalence margin과 충분한 반복을 사전 정의하지 않은 v1에서는
“관측된 seed 변동 범위에서 분류 성능이 비슷하다”까지만 기술합니다.

## Frozen spatial probe 계약

### Trimap 변환

공식 annotation README의 값을 다음과 같이 사용합니다.

| 원본 값 | 의미 | probe target |
|---:|---|---:|
| 1 | pet foreground | 1 |
| 2 | background | 0 |
| 3 | ambiguous boundary/accessory | 255, ignore |

경계는 foreground나 background로 임의 편입하지 않고 loss와 metric에서 모두
제외합니다. morphology, connected component, 수작업 수정과 결과 기반 표본 제외는
금지합니다.

`224 x 224` trimap을 nearest-neighbor로 변환한 뒤 `14 x 14` target을 만듭니다.
각 patch에서 값 3을 제외한 유효 픽셀만 분모로 사용하며 foreground 비율이 `0.5`
이상이면 foreground, 미만이면 background입니다. 유효 픽셀이 하나도 없는 patch는
ignore합니다.

### Encoder와 feature

- 분류 validation으로 선택한 checkpoint를 strict load
- encoder `eval()`, 모든 gradient 비활성화
- 마지막 block index `11`, `norm=False`, CLS token 제외
- feature `B x 192 x 14 x 14`, cache float32
- cache 생성 시 AMP 사용 안 함
- method별 encoder seed 3개를 서로 다른 cache로 보존

`norm=False`는 Phase 0/0.5에서 감사한 feature 계약을 유지하기 위한 선택입니다.
post-norm이나 중간 layer는 Phase 2 민감도 분석으로 분리합니다.

### Probe 학습

| 항목 | 고정값 |
|---|---|
| head | bias 포함 `Conv2d(192, 2, kernel_size=1)` |
| parameter | 386개 |
| 초기화 | weight `Normal(0, 0.01)`, bias `0` |
| loss | unweighted 2-class CE, ignore index `255` |
| optimizer | SGD, momentum `0.9`, weight decay `0`, Nesterov 없음 |
| LR grid | `[0.01, 0.03, 0.1]` |
| epoch | `100` |
| batch | `64` |
| schedule | cosine to `0` |
| probe seed | `[1,2,3,4,5]` |
| augmentation | 없음 |

같은 probe seed는 모든 encoder에서 byte-identical한 head 초기값과 batch 순서를
사용합니다. encoder checkpoint만 달라지고 probe 쪽 난수 조건은 달라지지 않습니다.

각 encoder checkpoint와 probe seed마다 세 LR을 모두 train split으로 학습합니다.
각 run은 validation `14 x 14` 2-class mIoU가 가장 높은 epoch를 보존합니다. 그중
validation mIoU가 가장 높은 LR을 선택하고, 동률이면 낮은 LR, 그다음 이른 epoch를
선택합니다. 선택을 마친 probe만 test에 한 번 적용합니다.

### Probe 평가

`14 x 14` logits를 `224 x 224`로 bilinear upsample하고 argmax합니다. 값 3이었던
경계 픽셀은 confusion matrix에서 제외합니다.

- 주 지표: `224 x 224` 2-class mIoU
- 보조 지표: foreground IoU, background IoU, foreground Dice, pixel accuracy,
  `14 x 14` mIoU
- 집계: 이미지별 metric 평균이 아니라 split 전체 global confusion matrix
- 비영상 baseline: all-background, train mean-mask

all-background는 모든 유효 pixel을 background로 예측합니다. train mean-mask는
train의 유효 `14 x 14` patch target만으로 각 위치의 foreground 빈도를 계산하고
(유효 target이 없는 위치는 0), 그 확률 map을 bilinear로 `224 x 224`에 올린 뒤
`>= 0.5`를 foreground로 예측합니다. 두 baseline 모두 test image RGB는 보지 않습니다.

정성 표본은 결과 전에 공식 test 순서의 거의 동일한 8개 quantile로 고정했습니다.

```text
Abyssinian_201
Bengal_33
chihuahua_67
great_pyrenees_91
miniature_pinscher_23
Ragdoll_37
shiba_inu_68
yorkshire_terrier_9
```

정성 표에는 encoder seed 1, probe seed 1을 사용하고 입력, GT, Vanilla, KD, LG,
ALG, iBKD 순으로 표시합니다. 사전 지정된 두 iBKD λ 중 결과가 좋은 값만 고르는
일이 없도록 iBKD-0.25와 iBKD-0.5 panel을 각각 생성하며, 공통 baseline 열은
동일하게 유지합니다. 이는 평가·선택 규칙을 바꾸지 않는 표시상의 명확화입니다.

## 반복과 통계 단위

한 encoder에 probe seed 5개를 붙여 얻은 5개 값은 서로 독립적인 encoder 반복이
아닙니다. 이를 15개 독립 run처럼 취급하지 않습니다.

1. encoder seed 하나 안에서 probe seed 5개의 mean ± sample SD를 계산합니다.
2. 방법별 주 결과는 encoder seed별 probe 평균 3개의 mean ± sample SD입니다.
3. 모든 encoder/probe seed 원값을 함께 공개합니다.
4. 주 paired contrast는 같은 encoder seed의 `iBKD - ALG`입니다.
5. encoder seed가 3개뿐이므로 형식적인 p-value를 주 결론으로 사용하지 않습니다.

Phase 1의 긍정적 신호는 iBKD–ALG mIoU 차이가 세 encoder seed에서 일관되게 양수이고
probe 초기화 변동보다 안정적인 경우입니다. 그렇지 않으면 Hold 또는 No-Go로
기록하고 결과를 본 뒤 v1의 metric이나 제외 규칙을 바꾸지 않습니다.

## 실행 gate

### H200 timing 전

- 아래 12-way timing matrix와 실행 코드를 commit하여 결과 이전 선택임을 보존

```text
(Vanilla, KD, LG, ALG, iBKD λ=0.25, iBKD λ=0.5) × batch (64, 128)
= student 12개 + timing 전용 teacher 1개
```

각 task는 전체 train 2,940장으로 실제 2 epoch 학습하고 validation 740장을 매
epoch 평가합니다. official test는 열지 않습니다. 기록하는 값은 epoch 시간,
300-epoch 환산시간, peak GPU memory, 성공/OOM 여부입니다. 2-epoch accuracy와
checkpoint는 과학 결과가 아니며 batch, λ, method 또는 checkpoint 선택에 사용할 수
없습니다. timing 전용 2-epoch teacher 역시 본 실험에 재사용하지 않습니다.

### H200 full classification 전

1. `images.tar.gz` SHA-256 기록
2. 7,349개 split ID의 image/label/trimap 1:1 감사 통과
3. `2,940/740/3,669` split manifest와 hash 생성
4. teacher/student/method loss 및 동일 초기화 단위 테스트 통과
5. 여섯 variant × 두 batch의 2-epoch full-data timing/smoke 완료
6. 예상 시간이 H200 요청의 Pod 제한 안에 들어가는지 확인
7. timing accuracy를 보지 않고 batch 64/128과 λ 0.25/0.5를 모두 보고하기로 LOCK

### H200 full classification 실행 분할

timing에서 얻은 300-epoch 환산시간만 사용하여 다음 두 요청으로 나눕니다.

| 요청 | 구성 | 예상시간(학습 환산) |
|---|---|---:|
| batch 64 | teacher 1 + 6 variants × 3 seeds = 19 tasks | 8h 39m 27s |
| batch 128 | teacher 1 + 6 variants × 3 seeds = 19 tasks | 8h 19m 51s |

두 요청은 독립 컨테이너이므로 같은 seed 1·batch 128 teacher를 각각 한 번
학습합니다. 두 결과의 `teacher_model_state_sha256`가 같아야 cross-profile 결과를
함께 해석합니다. 각 profile 안에서는 한 teacher checkpoint를 모든 guided method와
seed가 공유합니다. 두 요청 모두 600분 제한 안에 있으며 위 시간에 최초 설치·다운로드·
dataset audit와 각 checkpoint의 1회 test 시간이 추가됩니다.

이 두 요청은 37-way 분류까지만 포함합니다. frozen segmentation probe는 모든
분류 결과와 checkpoint 계약을 회수·감사한 뒤 별도 요청으로 실행합니다.

### Probe 전

1. 최종 고정한 방법·batch·λ × encoder seed 3개의 모든 분류 run 완료
2. validation-selected checkpoint strict load와 SHA-256 확인
3. official test가 selection에 사용되지 않았음을 summary로 확인
4. encoder frozen/gradient 0과 feature cache 계약 확인

batch 64 frozen probe 본 실험은 다음 명령으로 실행합니다.

```bash
bash phase1/scripts/run_probe_full_b64.sh
```

6 variants × 3 encoder seeds × 5 probe seeds × 3 LR의 270개 후보를 validation으로
선택한 뒤, 선택된 90개 probe 모두가 확정된 다음에만 official test를 엽니다.
test에서는 각 선택 probe를 정확히 한 번 forward하여 `14 x 14` 및 `224 x 224`
지표와 사전 고정 정성 예측을 함께 계산합니다.

Timing run은 시간·메모리·작업 분할 추정용이며 과학 결과나 checkpoint로 사용하지 않습니다. 이후
H200 요청 Issue에는 commit SHA, 실행 명령, 이미지, GPU 할당량, 예상 시간, output
경로와 회수할 artifact를 명시합니다.

## 결과 이후 변경 규칙

본 실험 protocol을 LOCK한 뒤에는 결과를 보고 split, augmentation, seed, boundary
처리, method coefficient, feature layer, normalization, LR grid, metric 또는 제외
규칙을 v1 안에서 바꾸지 않습니다.
변경이 필요하면 현재 실행을 pilot으로 표시하고 `v2` protocol을 만든 뒤 비교하는
모든 방법을 같은 새 조건으로 다시 실행합니다.

Phase 1이 지지할 수 있는 결론은 frozen feature의 **공간정보 선형 복원성** 차이입니다.
완성된 segmentation 성능, grid preservation의 단독 인과 효과 또는 표준 semantic
segmentation 우월성은 이후 Phase에서 별도로 검증합니다.
