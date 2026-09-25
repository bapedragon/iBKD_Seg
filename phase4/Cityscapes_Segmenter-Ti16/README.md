# Cityscapes · Segmenter-Ti/16

2026-09-26 사용자 결정: Tiny를 먼저 검사하며 **기존 OpenMMLab teacher를 유지**합니다.
이 폴더는 L/16 결과를 변경하지 않는 별도 실험입니다. 현재 구현 범위는 초기 손실 측정과
25-step 연결 smoke입니다. H200에서의 통과 결과나 500/2k/10k 결과는 아직 없습니다.
이슈는 사용자가 제출하며 이슈 입력용 MD 파일이나 GitHub 이슈는 생성하지 않습니다.

**현재 실행은 C2VKD*까지 추가한 v3(7경로)**입니다.
Vanilla, LG, ALG, iBKD λ0.25, iBKD λ0.5, FSKD*, C2VKD*를 순서대로 검사합니다.
고정 손실과 출처는 [FSKD_PROTOCOL.md](FSKD_PROTOCOL.md),
[C2VKD_PROTOCOL.md](C2VKD_PROTOCOL.md), 실행값은
[smoke25_fskd_c2vkd_v3.json](configs/smoke25_fskd_c2vkd_v3.json)에 있습니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_smoke25_fskd_c2vkd.sh
```

각 경로에서 무업데이트 calibration25 batch 후 상태를 복원하고 25 update를 합니다.
LG/ALG/iBKD의 β 4개는 **제안만 하고 최솟값 하나로** smoke합니다.
FSKD*와 C2VKD*는 고정 계수를 쓰며 β 탐색에 포함하지 않습니다.
FSKD*는 공개 DeiT-Ti 분류 예제를 segmentation에 이식한 구성입니다.
C2VKD*는 원본의 미공개 pooling 가중치를 CLIP RN101으로 대체하여 추가 사전학습
정보를 사용하므로 `supplementary_extra_pretraining` 결과로 별도 표시합니다.
두 방법 모두 저자 Cityscapes 설정의 완전 재현이라는 뜻은 아닙니다.

v3는 torchsort0.1.10 CUDA 확장을 설치하고 CLIP RN101 파일(291,791,292 bytes)을
다운로드·해시 검증해 **attention pool만** 사용합니다. OpenMMLab teacher는 유지합니다.
출력은 `/app/output/cityscapes_ti16_crop512_smoke25_fskd_c2vkd_v3/run_<UTC>_<PID>/`입니다.
`artifacts/smoke_summary.json`과 마지막 `[CITYSCAPES_TI16_SMOKE_FINAL]`에
7개 경로의 loss, 고정 계수/β 후보, 완료 step, 진단 pixel accuracy·mIoU·19 IoU,
재개 검사, teacher/pool 고정, 시간·메모리, 비교군 구분을 기록합니다.
**val 2장 점수는 연결 진단용이며 성능 순위를 정할 수 없습니다.**
통과 조건은 7경로 모두 passed 및 공통 초기화·입력 교차 검사 통과입니다.

이전 5경로 v1과 FSKD 추가 6경로 v2는 설정 파일과 실행 스크립트를 보존합니다.
아래의 5경로 상세 설명·기본 `run_smoke25.sh` 명령은 **v1 기록**입니다.
v2 전용 명령은 `run_smoke25_fskd.sh`이며, v2의 C2VKD 보류 표시는 과거 결정입니다.

## 고정 조건

- Teacher: 기존 MMSeg DeepLabV3 ResNetV1c-101-D8 Cityscapes 80k 체크포인트.
  348,988,299 bytes, SHA-256 `9e428899b279f29964cec79ab21bb19193328b8c4d42c0db49ff9070e9ab3b2d`.
  가중치와 BN 통계를 고정하며 CIRKD teacher로 교체하지 않습니다.
- Student: 공식 Segmenter `vit_tiny_patch16_384`, 12블록·192채널, patch16,
  mask-transformer decoder 1블록(폭 192). 깊이는 기존 L/16 비교 조건에서 가져왔습니다.
- Tiny 초기화: timm 0.4.12가 지정한 Google ViT AugReg ImageNet-21k→1k 사전학습 NPZ.
  23,226,422 bytes, SHA-256 `4b99893dc1a5a2a7d9ad119671c20559850f865e1fa17ed23401a3fefa7fedc9`.
  CNN으로 사전 증류한 DeiT checkpoint를 사용하지 않습니다. Tiny와 Large의 AugReg
  사전학습 세부 증강·정규화가 완전히 같다는 주장은 하지 않습니다.
- Cityscapes fine train 2,975 / val 500 / 19 class, test·coarse 미사용.
  원본 MMSeg augmentation·전처리와 L/16의 seed별 입력 순서를 재사용합니다.
- crop/window/stride 512, 실제 batch8, seed1, FP32, TF32/AMP/gradient clipping 없음.
- SGD Nesterov LR0.01, momentum0.9, WD0, poly power0.9, minLR1e-5,
  **80,000-step schedule의 처음 25 update**. LR warm-up 없음.
- LG/ALG: 0-based block `[0,6,11]`, iBKD: 12블록 전체 aggregation.
- ALG guidance warm-up0, iBKD warm-up20epoch, window50, threshold−0.02 유지.
  iBKD는 L/16에서 검증한 `flatmax_cpu_deform_v1` 연산을 재사용합니다.
- Smoke는 DataLoader worker0을 사용합니다. 원래의 sample별 독립 증강 seed와 순서는
  유지하지만 step 시간은 이 smoke 실행 조건의 측정값입니다.

이 설정은 **우리 L/16 crop512 조건을 Tiny로 이식한 실험**입니다.
저자가 공개한 Tiny 전용 Cityscapes 성능의 재현이나 Tiny의 최적 설정을 주장하지 않습니다.
공통 설정 원본은 [L/16 v19](../phase4_cityscapes/configs/paper_l16_crop512_final80000_v19.json),
이번 실행 값은 [smoke25_v1.json](configs/smoke25_v1.json)입니다.

## 이번 smoke의 범위

실행 경로는 Vanilla, LG, ALG, iBKD λ=0.25, iBKD λ=0.5의 5개입니다.

1. 각 경로가 같은 초기 student와 같은 train batch25개(200장)를 사용합니다.
2. Optimizer update 없이 train mode에서 픽셀별 CE와 raw guidance를 측정합니다.
   측정 후 모델·adapter buffer와 RNG를 복원합니다. Teacher는 계속 eval/frozen입니다.
3. `β₀ = 0.03 × median(CE) / median(guidance)`로 시작값을 계산하고
   `[β₀, 2β₀, 4β₀, 8β₀]`를 **제안 후보**로 기록합니다. 실제 batch별 가중 guidance/CE
   비율도 기록합니다. 이는 최적성이나 gradient 영향력 동등성을 보장하는 공식이 아닙니다.
4. 각 KD 경로는 β₀ 하나로 25 update, Vanilla는 CE만으로 25 update합니다.
   이 실행에서는 4개 β를 각각 학습하지 않습니다. λ별 β는 독립적으로 계산합니다.
5. 24번째 update 후 전체 상태를 저장한 뒤, 25번째 update를 재개해 모델·adapter·
   optimizer·schedule·입력을 비교합니다. 이는 1-update 재개 검사이며 장기 재현성 검사는 아닙니다.
6. 고정 val 2장을 원본 해상도에서 sliding window로 평가하고 pixel accuracy·mIoU·
   19 class IoU를 기록합니다. **진단 점수이며 β나 방법의 순위를 고르는 데 쓰지 않습니다.**
7. 방법 간 student 초기값·입력·teacher 일치, LG/ALG의 25-step 궤적 일치를 검사합니다.
   iBKD λ 두 조건의 adapter 초기값도 동일해야 합니다.

25 step은 한 epoch(372 step)를 채우지 않으므로 controller에 가짜 완료 epoch를
전달하지 않습니다. 자연스러운 guidance 종료는 이 smoke에서 검사하지 않습니다.
별도의 합성 loss 검사로 controller 경계·상태 복원을 확인하고 구분해 기록합니다.

## 실행과 결과

저장소 루트에서:

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_smoke25.sh
```

입력 ZIP은 `/app/data/chaoyang/`, 압축 해제 데이터는
`/app/scratch/cityscapes_l16_crop512_v3/cityscapes/`를 검증해 재사용합니다.
공식 소스·teacher 캐시는 `/app/scratch/cityscapes_official_l16_v2/upstream/`를 재사용하고
Tiny 초기 가중치만 추가합니다. 기존 L/16 asset 검증의 기본 동작은 유지합니다.

출력은 `/app/output/cityscapes_ti16_crop512_smoke25_v1/run_<UTC>_<PID>/`입니다.
재실행은 새로운 출력 폴더를 만들며 기존 결과를 덮어쓰지 않습니다.

- `artifacts/smoke_summary.json`: 다섯 경로와 교차 검사 결과, 가중치 해시.
- `artifacts/<run>/calibration.json`: 25개 초기 loss, 입력 해시, 후보 β와 실측 비율.
- `artifacts/<run>/summary.json`: update별 loss, 재개 검사, 진단 지표, 시간·메모리.
- 실패하면 `traceback.txt`, `training_progress.json`, `warnings.json` 등 남아 있는 진단 기록.

마지막 `[CITYSCAPES_TI16_SMOKE_FINAL]` JSON에 방법별 마지막 loss·SegLoss·guidance,
λ·실제 β·후보 β, 완료/선택 step·epoch, 진단 pixel accuracy·mIoU·클래스별 IoU(%),
teacher 고정·재개·입력 동일성, 시간·메모리를 함께 출력합니다. 평가하지 못한 값은 null입니다.
성공은 `status=passed`, 다섯 run 모두 passed 및 교차 검사 통과로 판단합니다.

## 실행 코드의 사전 검사

v3 추가 검사: 관련 단위 검사 총 45개, Python/shell 문법 및 설치 실패 시 종료 JSON 검사 통과.
공식 Tiny NPZ와 byte/SHA 검증한 실제 CLIP RN101 pool을 CPU에서 연결해, hook 전후
logit 일치·encoder/decoder/두 adapter gradient·pool 무변화 및 momentum/RNG를 포함한
저장/재개 후 동일 update 재현을 확인했습니다. 이 검사는 32×32 합성 입력과 합성 teacher
feature를 썼으며 H200/crop512/실제 teacher의 전체 실행 통과를 뜻하지 않습니다.


2026-09-26 로컬 검사: 관련 단위 검사 37개, Python 문법·shell 문법 검사 통과.
검증한 공식 Tiny NPZ를 로딩해 32×32 합성 입력에서 12개 feature와 segmentation
출력, CPU 학습 1 update 및 RNG·optimizer 복원 후 동일 update 재현을 확인했습니다.
실제 Tiny와 합성 teacher feature를 연결한 다섯 경로의 backward도 통과했습니다.
설치 단계 실패를 주입했을 때 후속 데이터 작업 없이 중단하고 마지막 JSON에 실패를
표시하는 것도 확인했습니다. **이는 H200/실제 Cityscapes smoke 통과를 뜻하지 않습니다.**

## 다음 단계

통과 후 β 제안값과 안정성 기록을 검토해 다음 500/2,000-step 후보 config를 고정합니다.
L/16에서 선정한 β를 Tiny의 검증값으로 사용하지 않습니다. 2k→10k→80k의 장기 재개
runner는 이 smoke에 포함되지 않으며, 준비되기 전 자동으로 실행하지 않습니다.
