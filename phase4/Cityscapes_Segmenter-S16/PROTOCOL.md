# Small 준비 프로토콜 v1

기준은 현재 Tiny/Large에서 사용하는 **Crop512 실험 프로토콜**입니다.
Segmenter 원 논문의 Cityscapes 성능을 그대로 재현한다고 표기하지 않습니다.
Small로 바꾸면서 모델 차원을 반영하고, 초기 loss 규모에 따라 β를 새로 측정합니다.

## 고정 조건과 근거

| 항목 | Small 고정값 | 기준 |
|---|---|---|
| Student encoder | `vit_small_patch16_384`, patch16, 12 blocks, width384, 6 heads | 고정 Segmenter 공식 config |
| Student 초기화 | 공식 AugReg ImageNet-21k → ImageNet-1k Small NPZ | timm0.4.12의 해당 모델 기본 URL |
| Decoder | mask-transformer 1 layer, 19 classes | 현재 Tiny/Large Crop512 실험 조건 |
| Teacher | OpenMMLab DeepLabV3-R101-D8, 공개 Cityscapes 80k checkpoint, frozen/eval | 기존 teacher 유지 |
| 데이터 | fine train2,975 / 공식 val500, ignore255, trainId0–18 | 기존 공식 split 및 label mapping |
| 입력 | image_size1024, random crop512×512, 기존 upstream 증강·정규화 | Tiny와 동일한 loader |
| Batch/seed | 단일 GPU batch8, 마지막 batch7 포함, seed1 | Tiny와 동일, epoch당372 updates |
| Optimizer | SGD Nesterov, LR0.01, momentum0.9, weight decay0 | 기존 Segmenter 경로 |
| LR schedule | poly power0.9, minLR1e-5, 전체 horizon80,000step, LR warm-up0 | 짧은 후보 실험도 같은 80k schedule의 앞부분 |
| 연산 | FP32, TF32 off, activation recomputation, gradient clipping 없음 | 현재 Tiny 조건 |
| 초기화/입력 통제 | student seed1, guide seed1001, 학습 RNG2001; epoch/sample별 증강 RNG | 기존 smoke의 공통 초기값/전체 입력 hash 검사 |
| iBKD | CPU deform 결정성 경로 `flatmax_cpu_deform_v1`, CPU thread1 | Tiny에서 검사한 구현 유지 |
| ALG | window50, threshold−0.02, controller warm-up0 | 사용자 결정 유지 |
| iBKD controller | window50, threshold−0.02, warm-up20epoch, λ0.25와0.5 | 사용자 결정 유지 |
| 평가 | 원본 해상도 복원, window512/stride512, single-scale, window batch1 | 현재 공통 평가 경로 |
| 지표 | accuracy 우선, 같은 checkpoint의 mIoU·19 class IoU 병기 | 사용자 결정 유지 |

공식 S/16 config와 초기화 근거:
[Segmenter config, 20d1bfa](https://github.com/rstrudel/segmenter/blob/20d1bfad354165ee45c3f65972a4d9c131f58d53/segm/config.yml),
[timm v0.4.12 ViT](https://github.com/huggingface/pytorch-image-models/blob/v0.4.12/timm/models/vision_transformer.py).
이름의 `384`는 공개 encoder 사전학습 설정이며 현재 학습 crop은512입니다.
위치 embedding은 원본 loader가 crop512에 맞춰 보간합니다.

Tiny의 공개 NPZ는 `aug_none`, Small은 `aug_light1`입니다.
둘 다 공식 ImageNet-21k→1k 초기화지만 encoder 사전학습 augmentation까지 동일하지는 않습니다.
Small 안의 방법 비교는 모두 **같은 Small NPZ와 decoder 초기값**을 사용합니다.

## 가중치와 소스 식별

Small encoder `vit_small_384.npz`:

- bytes: `88851254`
- SHA256: `9e4155229ce0b767ef43ff1e3a9f4308a69b61f44969e0605df5f39047f2e7a8`
- 원본: [공식 Small AugReg NPZ](https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz)

Teacher `deeplabv3_r101.pth`:

- bytes: `348988299`
- SHA256: `9e428899b279f29964cec79ab21bb19193328b8c4d42c0db49ff9070e9ab3b2d`
- 기존 `deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth` 유지.

Segmenter `20d1bfad354165ee45c3f65972a4d9c131f58d53`,
MMSegmentation `26032167e005f391aad1bf151a7e8f9a98973266`,
MMCV `db097bd1e97fc446a7551c715970611d2fcc848d`, timm0.4.12를 사용합니다.
Tiny/Large 때와 같이 고정 소스의 commit·변경 여부와 가중치 byte/SHA를 검사합니다.
원본 ZIP SHA는 공식 배포 SHA라고 주장하는 값이 아니라 사용자가 전달한 파일의 식별값입니다.
실제 소스·가중치 provenance는 각 결과 폴더에 기록합니다.

## β 후보 생성

LG/ALG, iBKD λ0.25, iBKD λ0.5 각각 **첫 25개 train batch에서 update 없이** 측정합니다.
`S = median(segmentation CE)`, `G = median(unweighted guidance)`라 두면:

```text
beta_base = 0.03 × S / G
beta_candidates = beta_base × [0.5, 1, 1.5, 2, 3, 4, 6, 8]
초기 목표 비율 = [1.5%, 3%, 4.5%, 6%, 9%, 12%, 18%, 24%]
smoke 학습 beta = beta_base (목표 3%, 두 번째 후보)
```

CE는 정답 segmentation mask와 예측 간 pixelwise cross entropy이며 void255를 제외합니다.
iBKD의 guidance는 `(1−lambda) × alignment + lambda × fusion`을 먼저 계산한 값입니다.
λ와 β는 다른 파라미터입니다. 각 λ에서 측정한 G가 다르면 후보 β도 다릅니다.

이 비율은 기존 Tiny의 넓은 초기 세기 탐색 기준을 유지한 **heuristic**입니다.
논문이 정한 최적 비율이나 gradient 기여도 비율이 아니며 학습 중 유지되지 않습니다.
median(CE)/median(guidance)로 정한 목표와 batch별 실제 비율의 median도 다를 수 있어
각 후보의 실제 min/median/max 비율을 따로 기록합니다. val은 이 산출에 사용하지 않습니다.

ALG는 같은 실행에서 LG가 산출한 **정확히 같은 β 후보와 smoke β**를 읽습니다.
student·adapter·teacher 초기 상태, 모든 calibration 입력, config·소스 hash가 다르면 거부합니다.
ALG 자체의 CE/guidance도 측정·보존해 LG와 비교합니다.
calibration 후 모델/adapter/RNG를 복원하고 train 0step부터 시작합니다.

## 비교군의 의미

- **Vanilla**: 같은 Small encoder+decoder를 segmentation CE만으로 학습.
- **LG/ALG**: 0,6,11번 encoder block을 사용하는 기존 locality guidance.
  ALG만 기존 epoch 단위 종료 controller를 사용.
- **iBKD λ0.25/λ0.5**: 12개 encoder block의 aggregation과 기존 alignment/fusion.
  teacher stage와 loss reduction은 Tiny와 동일, 입력 채널을384로 변경.
- **FSKD* (Tiny recipe transferred to S/16)**: 기존 Tiny의 DeiT-Ti 분류 예제 기반
  segmentation adaptation을 유지. student block0/3, teacher stage0/1,
  CE1·logitKD1·global100·patch1·attention1e6. adapter 입력 채널384, CLS attention6heads.
  저자가 검증한 Small/Cityscapes recipe가 아니며 계수의 최적성을 주장하지 않음.
- **C2VKD* (CLIP-pool)**: 기존 Tiny 구현에서 adapter 채널만384로 변경.
  PDD1·global0.1·patch0.1·linguistic0.5, standalone CE0(CE는 진단).
  원본 미공개 pool 가중치를 CLIP RN101으로 대체하고 공개 코드의 누락 loss를 재구성한
  비교군이므로 원본 완전 재현으로 표기하지 않음. 추가 CLIP 사전학습을 쓰는 보조 비교군으로 분리.

FSKD 소스 commit `969dddf278b2c9f2dadde504326fa9d704c5a5aa`,
C2VKD 소스 commit `fe1ab3d6f815969058451c221229a744fe872a47`과 세부 adaptation은
고정 JSON의 `fskd`, `c2vkd` 항목에 기록합니다.
FSKD*/C2VKD*는 별도 β를 산출하지 않습니다.

## 해석 범위

첫 smoke는 25step endpoint에서 고정 val2만 평가합니다. 성능 순위나 자연스러운 가이던스
종료를 검사하지 않습니다. 저장 재실행은 단일 update의 근사 일치 검사이며
두 개의 긴 독립 학습 전체가 재현된다는 뜻이 아닙니다.
CPU에서 model 로딩이 통과해도 H200 경로·Small β별 안정성은 새로 확인해야 합니다.
Tiny 작업을 마친 후 시작하는 순서와 단계별 계획은 [README](README.md)를 따릅니다.
