# SegFormer-B0 공통 프로토콜 v1

2026-09-23 사용자 결정에 따라 **FSKD의 segmentation 명시값 → CIRKD의 해당 모델 공개 코드
→ SegFormer 공식 Cityscapes B0 설정** 순서로 공통 프로토콜을 고정했습니다.
식별자는 `cityscapes_deeplabv3r101_segformerb0_common_v1`입니다.

- [값·출처·예외 판단을 담은 프로토콜 명세](configs/segformer_b0_common_protocol_v1.json)
- [비교 조건·확정 범위 명세](configs/comparison_manifest_v1.json): 공통 JSON의 SHA-256,
  비교할 여섯 조건, β 선별 규칙과 방법별 미확정 항목을 연결합니다.
- 상태: **공통 조건 고정 / H200 smoke·25-batch β 측정 통과 / pack1·2 후보 결과 확인, pack3 미확인**
- 이 JSON은 실행용 config가 아닙니다. 기존 L/16 runner의 config로 넣지 않습니다.
- 완료된 L/16 2k·10k 결과는 [기존 실험 폴더](../phase4_cityscapes/README.md)에 보존합니다.
  B0의 성능이나 최적 beta로 해석하거나 B0 학습의 초기 가중치로 사용하지 않습니다.

여기서 채택한 것은 공개 근거를 우선순위대로 조합한 **우리의 공통 비교 조건**입니다.
FSKD의 누락값까지 저자 설정으로 확인했다는 뜻은 아닙니다.

현재 실행 revision은 [2k v2 명세](configs/b0_screen2000_v2.json)입니다.
Smoke/calibration에서 검증한 NVIDIA 공식 HF ImageNet-only MiT-B0 역변환을 모든 비교
조건에 동일하게 사용하도록 출처 예외를 명시했습니다. CIRKD Baidu 원본과의 동일성은
미확인입니다. 아래 원래 v1의 출처·JSON 해시는 보존하며 이 예외를 숨기지 않습니다.
실행 명령과 내장 검사는 [세 H200 이슈](H200_SCREEN2000_ISSUES.md)에 있습니다.
v1 pack1은 모두 25 update 후 개별 ignore-only crop 거부 조건에서 실패했습니다.
v2에서는 CIRKD처럼 crop을 유지하고 기존 CE·logit KD가 배치 전체 유효 픽셀을 계산합니다.
재표집·배치 축소는 없고 feature guidance도 원래 배치에 적용합니다.
배치 전체에 유효 라벨이 없으면 명시적으로 실패합니다. β·LR·입력 난수 계획은 유지하며,
v1 checkpoint와 섞지 않고 모든 후보를 seed1부터 실행합니다.

## 이번에 확정한 비교 범위

2026-09-23 재확인에서 공통 JSON의 값은 유지했습니다. 비교 대상은 **FSKD, LG, ALG,
iBKD λ=0.25, iBKD λ=0.5**이며, 증류 효과를 확인할 **Vanilla**를 함께 둡니다.
모든 KD 조건의 teacher 파일·student 초기화·입력·학습 예산·평가 방식을 공유합니다.
CIRKD는 이번에는 공통 설정과 구현의 출처이며, CIRKD 증류 loss를 비교 조건에 추가하지 않습니다.

공통 학습·평가 조건과 비교 범위는 확정됐습니다. 후속 논의에서 iBKD 가중합 전
목표 규격은 **256채널·16×16**으로 선택해 별도 비교 명세에 기록했습니다.
이어 LG/ALG의 **처음·중간·끝인 1·5·8번째 block**과 controller의
**1 epoch 분량(186 step)마다 관측**을 사용자 결정으로 고정했습니다.
기존 window 50, threshold -0.02는 유지하고, 후속 사용자 결정에 따라 guidance warm-up은
**ALG 0·iBKD 0 epoch**로 설정합니다. iBKD 두 lambda 조건에 동일하게 적용합니다.
구체적인 종료 경계·중단 구간 처리와
전체 8개 block 연결은 별도 비교 명세에 기록했습니다.
후속 요청에 따라 FSKD와 C2VKD의 [재구현 프로토콜](BASELINE_METHOD_PROTOCOLS.md)을 작성했습니다.
FSKD의 첫 연결·계수는 선정했고, 저자 Cityscapes 값의 미확인 여부는 별도로 표시했습니다.
수치 β 후보는 25-batch 측정으로 고정했고 H200 smoke도 통과했습니다.
2k v2 pack1은 GPU 2k·전체 val500을 완료했고 LG의 10k 후보 2개를 선정했습니다.
Pack2 후보 기록도 확인해 ALG와 iBKD λ=0.25의 10k 후보를 각각 2개 선정했습니다.
Pack2는 첨부 group header가 없어 후보별 기록으로 선별을 재계산했다는 한계를 남깁니다.
Pack3 결과와 최종 β 선정은 아직 확인 전입니다.
따라서 **공통 프로토콜 확정**과 **모든 방법의 실행 준비 완료**를 구분합니다.

C2VKD 대체안은 기존 6개 비교 밖의 추가 후보입니다. 원본 pooling 가중치가 없어
CLIP pool을 대체 사용하는 추가 사전학습 조건을 명시했습니다. 또한 C2VKD의
PDD가 지도학습을 포함하므로 해당 방법 JSON만 CE 계수를 0으로 override합니다.
공통 JSON과 기존 6개 방법의 CE 계수 1은 바꾸지 않습니다.

## 출처 버전과 적용 순서

1. **FSKD**: [SSRN preprint](https://ssrn.com/abstract=5385770)의 4.1절, Table 1,
   Figure 7. 확인한 `ssrn-5385770.pdf`의 byte size와 SHA-256을 명세에 기록했습니다.
   segmentation에서 명시된 조건을 우선합니다.
2. **CIRKD**: [2022-10-21 공개 코드](https://github.com/winycg/CIRKD/tree/48eb81b7a0ed7c59f9e347e97feb95a23e943d12)
   `48eb81b7a0ed7c59f9e347e97feb95a23e943d12`로 고정합니다. 학생은 SegFormer의
   학습·모델·평가 경로를, CNN teacher는 DeepLabV3-R101 구현과 가중치를 사용합니다.
   현재 main의 CIRKDV2나 원 논문의 CNN 학생용 SGD를 섞지 않습니다.
3. **SegFormer 공식 B0**: [NVlabs Cityscapes config](https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/local_configs/segformer/B0/segformer.b0.640x1280.city.160k.py),
   `65fa8cfa9b52b6ee7e8897a98705abf8570f9e32`입니다. 상위 자료에 실제로 빈 항목이
   남을 때만 보완합니다. 이번 주요 항목은 1·2차 자료에서 결정됐으며, 공식 설정에서는
   MiT-B0 구조·ImageNet 초기화·AdamW betas의 일치를 추가 대조했습니다.

## 고정된 공통 조건

| 항목 | 고정값 | 채택 근거 |
|---|---|---|
| 데이터 | fine train 2,975장, val 500장, 19개 class | FSKD Table 1 + CIRKD split/label 코드 |
| 데이터 사용 범위 | coarse·train_extra 미사용, test는 학습·선택에 미사용 | 위 fine train/val 실험 범위 |
| Teacher | DeepLabV3-ResNet101 | FSKD 4.1 |
| Teacher 가중치 | CIRKD의 Cityscapes teacher, 공개 보고 val mIoU 78.07 | CIRKD README; FSKD와 동일 파일인지는 미확정 |
| Teacher 상태 | 전체 freeze, eval mode, BN 통계 고정 | CIRKD 학습 코드 |
| Student | SegFormer MiT-B0, CIRKD 공개 구현 | FSKD 모델 선택 + CIRKD 구현 |
| Student 초기화 | ImageNet pretrained `mit_b0.pth` + 무작위 segmentation head | CIRKD B0 실행 스크립트 |
| 학습 crop | 512×512 random crop | FSKD 4.1 |
| Batch | 총 16, optimizer update당 16장 | FSKD 4.1 |
| 학습 길이 | 80,000 optimizer step | FSKD 4.1 |
| Optimizer | AdamW, betas=(0.9, 0.999), eps=1e-8 | CIRKD SegFormer + 해당 PyTorch 기본값; betas는 NVlabs와도 일치 |
| 초기 LR | 6e-5 | FSKD 4.1 |
| Weight decay | 1e-4 | CIRKD SegFormer 학습 코드 |
| LR 감소 | `6e-5 × (1-k/80000)^0.9`, k=완료한 update 수 | power는 FSKD, 적용 시점은 CIRKD 코드 |
| LR warm-up | 0 step | CIRKD 코드에 warm-up 없는 poly schedule |
| 파라미터별 LR·WD | 동일 LR/WD, head LR×1, norm/bias의 WD 면제 없음 | CIRKD optimizer 단일 group |
| 배율 증강 | 원본 1024×2048에 0.5~2.0 배율, 0.1 간격 16개 중 균등 선택 | CIRKD `dataset/cityscapes.py` |
| 좌우 반전 | 확률 0.5 | CIRKD dataset |
| 색상 증강·class 비율 제한 crop | 사용하지 않음 | CIRKD dataset의 실행 경로 |
| 전처리 | BGR, 0~255, mean=[104.00698793,116.66876762,122.67891434] 차감, std 나눔 없음 | CIRKD dataset |
| Decoder | 4개 stage MLP 결합, embedding 256, dropout 0 | CIRKD `models/segformer.py`의 B0 |
| Encoder dropout / drop-path | dropout 0, attention dropout 0, drop-path 최대 0.1 | CIRKD B0 |
| 공통 지도학습 loss | 유효 pixel 평균 CE, 계수 1, ignore=-1, OHEM·aux loss 없음 | CIRKD SegFormer/task loss |
| 정밀도·gradient clipping | FP32, clipping 없음 | CIRKD 학습 경로 |
| 평가 시점 | 매 400 step, 마지막 80,000 step 포함 | FSKD Figure 7 |
| 평가 범위 | 매번 val 500장 전체, single-scale 1.0, flip 없음 | CIRKD 평가 코드 |
| Student 추론 | 원본을 좌우 1024×1024 두 영역으로 나눠 추론, logits를 이어 붙인 뒤 원본 크기로 보간 | CIRKD 독립 `eval_segformer.py` |
| Checkpoint 선택 | val mIoU 최대, 동점이면 앞선 step | CIRKD best-pred 비교 |
| 함께 기록할 지표 | 선택 checkpoint의 pixel accuracy·mIoU·19 class IoU 및 마지막 step 지표 | 공통 보고 규칙 |

평가 보간은 logits에 bilinear `align_corners=True`를 적용한 뒤 argmax합니다.
MiT 내부의 다중 stage feature 결합 보간은 CIRKD 모델에 따라 `align_corners=False`입니다.
Train/val의 non-evaluation label은 모두 -1로 바꾸고 loss와 metric에서 제외합니다.
Teacher 기준 성능 확인은 CIRKD CNN 평가 방식인 원본 전체 이미지 추론으로 별도 기록합니다.

## 혼동하기 쉬운 선택의 근거

- **초기화:** FSKD 공통 Algorithm 1의 random student 문구만으로 Cityscapes MiT encoder의
  초기화를 확정하지 않았습니다. 앞선 감사에서 segmentation 미확정으로 분류한 항목이므로
  2순위인 CIRKD의 ImageNet pretrained 설정을 채택했습니다. 이를 FSKD가 명시한 값으로
  인용하지 않습니다.
- **추론:** CIRKD의 guided 내부 `validation()`에는 전체 이미지 경로가 있고,
  독립 `eval_segformer.py`와 baseline validation에는 좌우 분할 경로가 있습니다.
  새 비교에서는 독립 평가 코드의 좌우 분할을 checkpoint 선택부터 최종 보고까지
  Vanilla/FSKD/LG/ALG/iBKD의 모든 조건에 동일하게 적용합니다. 평가 때 512 crop을
  자동 사용하거나 이미지 전체를 512로 축소하지 않습니다.
- **3순위 설정:** NVlabs의 WD=0.01, warm-up 1,500, head LR×10, RGB/std 정규화,
  decoder dropout 0.1, photometric distortion은 CIRKD에서 이미 값이 정해진 항목이므로
  가져오지 않았습니다. AdamW betas도 CIRKD가 사용하는 PyTorch 기본값과 이미 일치합니다.
- **선택 metric:** 새 B0 프로토콜은 위 우선순위에 따라 mIoU로 checkpoint를 고릅니다.
  기존 L/16의 pixel accuracy 우선 선택 규칙은 해당 L/16 결과에만 적용됩니다.
- **Metric 구현:** CIRKD `utils/score.py`의 pixel accuracy에는 logits를 정수로 바꾼 뒤
  argmax하는 경로가 있습니다. 새 구현에서는 float logits의 argmax로 올바른 pixel
  accuracy를 계산합니다. 논문 설정을 채택하는 것과 metric 코드의 오류를 복사하는 것은
  별개이며 이 보정은 모든 방법에 공통 적용합니다.

## 운영 결정과 다음 구현

첫 실행은 기존 환경에 맞춰 **seed1, H200 1장, 실제 batch16, accumulation 1**로
고정합니다. seed와 GPU 장수는 FSKD가 명시한 값으로 주장하지 않습니다. 데이터 순서는
CIRKD의 train list를 80,000×16 길이로 반복·잘라 만든 후 shuffle하는 방식을 사용하고,
모든 방법에서 같은 student 초기 상태·입력 순서·crop·flip을 공유합니다.
방법 전용 난수와 입력 난수는 분리하고, TF32는 끕니다.

가중치 실물의 byte size/SHA-256과 strict 적재는 [asset 명세](configs/b0_asset_sources_v1.json)에
기록했습니다. NVlabs Drive 404로 smoke student는 NVIDIA 공식 HF의 ImageNet-only MiT-B0를
역변환합니다. CIRKD Baidu 원본과의 동일성은 미확인입니다. 공통 JSON의 원래 해시는
유지하며, 초기 smoke 예외를 [2k 실행 명세](configs/b0_screen2000_v2.json)에 명시적으로 승계했습니다.
3 update·val2·checkpoint 재실행과 실제 batch·메모리·입력 동일성은 H200 smoke v2에서
통과했습니다. 새 [2k 이슈](H200_SCREEN2000_ISSUES.md)에는 전체 val500과 내장 재개 검사가 있습니다.
10시간 작업 제한은 9시간 실행 후 상태 저장·동일 schedule 재개로 처리합니다.

**방법별 설정은 이번 공통 프로토콜에 넣지 않습니다.** FSKD의 loss 가중치·KD
temperature/alpha·feature pairing, CIRKD의 memory/relational loss, LG/ALG/iBKD의
beta·lambda·controller·guidance warm-up은 별도 방법 명세에서 정합니다.
공통 LR warm-up 0과 방법별 guidance warm-up은 다른 항목입니다.

비교할 방법, 후보 선별 순서와 방법별 미확정 항목은 [실험 흐름](EXPERIMENT_PLAN.md)에
별도로 정리합니다. 이 문서와 JSON은 공통 조건의 근거이며, 방법 설정의 확정 여부는
[폴더 안내](README.md)에서 함께 확인합니다.
