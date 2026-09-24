# SegFormer-B0 비교 실험 흐름

현재 우선 실행은 **λ=0.25·warm-up 20의 β4개 짧은 smoke → 같은 네 β를 각각 2k 재실행**입니다.
[지금 제출할 smoke 이슈](H200_WARMUP20_SMOKE_ISSUE.md)를 먼저 사용하고 결과 확인 뒤 후속 실행을 진행합니다.

**2026-09-24 최신 변경:** iBKD λ=0.25·0.5에만 guidance warm-up 20을 적용하고 ALG는 0으로
유지합니다. [변경 프로토콜](WARMUP20_PROTOCOL.md) 및 [이슈 2개](H200_WARMUP20_ISSUES.md)를
준비했습니다. 기존 2k 결과·고정 명세는 warm-up 0 기록으로 보존합니다.
새 iBKD는 λ별 β 4개를 초기화부터 10k까지 비교하며 2k에서는 후보를 제외하지 않습니다.

작성: 2026-09-23. 계획 ID: `cityscapes_segformer_b0_comparison_plan_v1`.
공통 조건의 기준은 [프로토콜](PROTOCOL.md)과
[JSON 명세](configs/segformer_b0_common_protocol_v1.json)입니다. 비교 범위와 현재 확정 상태는
[비교 명세](configs/comparison_manifest_v1.json)에 기록했습니다.
이 문서는 실행 순서와 선별 방식의 계획이며 GPU 결과가 아닙니다.
후속 작업으로 7개 방법이 [H200 smoke v2](reports/h200_smoke_v2/RESULTS.md)를 통과했습니다.
[25-batch calibration](reports/h200_beta_calibration_v1/RESULTS.md)도 통과해 β 후보를 고정했습니다.
다음 단계의 [2k 이슈 3개](H200_SCREEN2000_ISSUES.md)와
[실행 명세 v2](configs/b0_screen2000_v2.json)를 준비했습니다.
첫 H200 2k v1 pack1의 6개 실행은 모두 25 update 후 개별 ignore-only crop 거부로 실패했습니다.
[실패 기록](reports/h200_screen2000_v1_pack1_failure/RESULTS.md)을 보존하며, 데이터 로더를 수정한
v2에서 모든 후보를 seed1부터 실행합니다. β·입력 순서·LR·평가·선별 규칙은 유지합니다.
2026-09-24 [2k v2 pack1 결과](reports/h200_screen2000_v2_pack1/RESULTS.md)를 확인했습니다.
6개 모두 완료했고 LG의 10k 후보는 β=0.197479·0.460784입니다.
Vanilla·고정 FSKD도 기존 설정으로 계속합니다.
이어 [pack2 후보별 기록](reports/h200_screen2000_v2_pack2/RESULTS.md)을 확인했고, ALG는
β=0.197479·0.460784를 유지합니다. 당시 iBKD λ=0.25는 β=0.903048·3.87021을 선정했습니다.
Pack2 원본 group header는 첨부에서 잘려 후보별 점수로 순위를 재계산했습니다.
[Pack3 결과](reports/h200_screen2000_v2_pack3/RESULTS.md)는 4/4 완료이며
iBKD λ=0.5는 당시 β=0.4303·1.00403을 선정했습니다.
기존 10개 재개 계획은 새 iBKD revision에서 변경합니다. 현재 10k 대상은 기존 Vanilla·FSKD
각 1개, LG·ALG 각 2개 및 새 iBKD 각 λ 4개로 총 **14개 실행 궤적**입니다.
10k·80k 결과는 아직 없습니다.

후속 요청에 따라 [FSKD·C2VKD 방법 명세](BASELINE_METHOD_PROTOCOLS.md)와 각각의 JSON을
작성했습니다. FSKD는 공개 PiT 조합을 이식한 고정값 1개로 아래 흐름에 연결합니다.
C2VKD는 원본 pooling 가중치 누락 때문에 CLIP pool 대체안까지 작성했으며,
추가 사전학습 조건을 가진 별도 후보로 관리합니다. 아래 기존 6개 조건의 후보 수는 유지합니다.

## 1. β 후보를 B0에서 다시 찾는 이유

기존 L/16과 B0는 encoder 폭·깊이·feature 해상도가 다릅니다. 새 공통 프로토콜은
teacher 가중치 출처, optimizer, batch와 전처리도 기존 L/16 실험과 다릅니다.
따라서 같은 β라도 CE 대비 guidance 크기와 student에 전달되는 gradient가 달라질 수 있습니다.
**L/16에서 잘 나온 β를 B0의 최적값으로 고정하지 않고, B0에서 다시 측정·선별합니다.**

현재 구현의 iBKD 손실 구조는 다음과 같습니다.

```text
Vanilla: L = CE
LG/ALG:  L = CE + β(t) × G_locality
iBKD:    L = CE + β(t) × [(1−λ) × G_alignment + λ × G_fusion]
```

LG의 β는 일정하고 ALG/iBKD의 β(t)는 controller 종료 후 0이 됩니다.
iBKD λ=0.25와 λ=0.5는 별도 비교 조건이며, **각 λ에서 β를 독립적으로 선별**합니다.
손실의 기준은 기존 [Cityscapes runner](../../src/ibkd_seg/cityscapes/official_full.py)와
[guidance 구현](../../src/ibkd_seg/phase1/models.py)입니다.

FSKD에는 구조 손실 세 가지와 logit KD가 있으므로 동일한 단일 β 문제로 취급하지 않습니다.
FSKD 고유 손실은 FSKD에만 적용하고, LG/ALG/iBKD에 자동 추가하지 않습니다.
공통 CE 계수 1과 FSKD 원문의 CE/KD 계수 표현을 어떻게 대응시켰는지도 방법 명세에 기록합니다.
**FSKD는 재구현 명세의 고정값으로 먼저 실행합니다.** 저자 Cityscapes 설정을 추후 확보하면
출처와 차이를 기록하고 별도 revision으로 반영합니다. 가중치 재탐색은 필수 단계가 아닙니다.

## 2. 후보 탐색 전에 완료할 준비

| 준비 항목 | 완료 기준 |
|---|---|
| 데이터·가중치 | 기존 ZIP 검증 결과 연결, CIRKD teacher와 MiT-B0 실제 파일의 bytes/SHA-256, key·shape 적재 검사 |
| 공통 runner | 실제 batch16, 공통 전처리·증강·80k LR·좌우 분할 val 평가, mIoU와 pixel accuracy 검증 |
| LG/ALG 연결 | 확정된 1·5·8번째 block 선택·기존 채널 투영·공간 정렬을 두 방법에 동일하게 구현·검증 |
| iBKD 연결 | 전체 8개 B0 block을 256채널·16×16로 맞춰 aggregation에 전달; λ 두 조건은 이 구현을 공유 |
| FSKD 연결 | 방법 명세의 stage 3·4, lg/cg 정렬, 평균 incoming attention, 계수와 reduction 구현·검증 |
| C2VKD 추가 후보 | 별도 명세의 PDD·feature 항·pool 적재 검증; CE 예외와 추가 사전학습 조건 표시 |
| Controller | 186-step 관측·raw guidance 평균·기존 window/threshold 유지. 기존 두 방법 warm-up 0, 새 revision은 iBKD만 20·ALG 0 |
| 비교·재개 | 같은 seed의 student 초기 상태·입력·증강을 공유; adapter 초기화와 optimizer/RNG/controller/sampler를 포함한 재개 확인 |

B0는 4개 stage에서 폭·해상도가 달라집니다. 기존 L/16의 24개 동일 폭 block을 가정한
LG feature index와 iBKD aggregation을 그대로 사용할 수 없습니다. 단순히 모델 이름만
바꾼 상태에서 β sweep을 시작하지 않습니다. Adapter 파라미터의 optimizer 포함 여부와
LR/WD도 방법 명세에 남깁니다.

### 2026-09-23 후속 연결 상태 점검

사용자는 iBKD의 가중합 전 목표 규격을 **256채널·16×16**으로 선택했습니다.
`16`은 공간 격자의 한 변이며 입력의 1/16 축소비율을 뜻하지 않습니다.
512×512 crop에서 이 규격은 B0 마지막 stage의 채널·공간 크기와 같습니다.
두 iBKD lambda 조건은 이 목표 규격을 공유합니다. 공통 backbone·decoder 설정의
변경이나 LG/ALG/FSKD의 feature 규격 변경으로 확대 해석하지 않습니다.

이전 CUB 분류 실험의 PVTv2·PiT·CvT에서는 이미 block별 채널·공간 크기가 달랐습니다.
기존 [전체 block adapter](../../../IBAM_KD_H200_V2/methods/table1_cub200/adapters.py)는
각 block을 192채널·14×14로 변환한 뒤 iBKD 가중합에 전달했습니다.
따라서 B0도 전체 8개 block을 추출하고 이 adapter의 목표를 256채널·16×16으로 바꾸는
방식을 사용합니다. 이 연결 설정은 B0 실행 검증 완료를 뜻하지 않습니다.
기존 방식은 채널이 다를 때 block별 학습 가능한 1×1 projection, 공간이 다를 때
bilinear resize(`align_corners=False`)를 사용합니다. 실제 feature 추출 위치와 teacher
대응, adapter의 optimizer 연결까지 B0 방법 명세와 구현에서 확인해야 합니다.

기존 iBKD alignment는 가중합 이후 teacher와 student 중 더 큰 격자에 맞춥니다.
이를 유지하면 16×16에서 다시 확대될 수 있으므로, 이번 결정은 fusion 전체를
16×16에서 계산하겠다는 결정이 아닙니다. 초기 stage의 작은 구조가 축소 과정에서
손실될 가능성과 전체 메모리 감소량은 실제 구현·실험에서 확인합니다.

| 점검 항목 | 현재 상태 |
|---|---|
| Teacher·student·공통 학습/평가·6개 비교 조건 | 기존 확정값 유지 |
| iBKD 가중합 전 채널·공간 크기 | 첫 실험값 256채널·16×16 선택 완료 |
| iBKD 전체 block 연결 | 전체 8개 block을 256채널·16×16로 변환 후 학습 가능한 가중합; H200 smoke v2 통과 |
| LG/ALG block 대응 | 사용자 결정으로 처음·중간·끝인 1·5·8번째, 코드 인덱스 0·4·7 확정 |
| ALG/iBKD controller | 186-step 관측, window50·threshold−0.02 유지. 9/23 두 방법 warm-up 0 → 9/24 iBKD만 20으로 별도 재실험 |
| iBKD λ와 β | 비교 λ=0.25·0.5 유지. 수치 β는 연결 후 측정·짧은 학습으로 선별 |
| FSKD 고유 설정 | 첫 재구현값 선정 완료; 저자 Cityscapes 값과의 동일성은 미확인 |
| C2VKD 고유 설정 | PDD 보완·공개 loss 연결·CLIP pool 대체안 작성; 기존 6개 밖의 별도 후보 |
| 실제 실행 준비 | 가중치 실물·CPU 연결/재개, H200 smoke·calibration 통과; 2k v2 세 묶음 결과 확인·10k 후보 선정; pack2 원본 집계 미확인 유지 |

LG/ALG는 후속 사용자 지시 "이번에도 처음 중간 끝"에 따라 기존 L/16의
`0, depth//2, depth-1`을 적용해 코드 인덱스 **0·4·7**을 사용합니다.
과거 PVTv2 분류의 0·3·7과는 구분하며, LG 저자의 공식 MiT-B0 설정이라고 주장하지 않습니다.
선택한 학생 출력 세 개는 기존 순서의 teacher 증류 feature 세 개와 대응합니다.
각 출력에 teacher 채널로의 1×1 projection을 적용하고 양쪽 중 더 큰 격자에
bilinear(`align_corners=False`)로 정렬한 뒤 세 mean MSE를 합하는 기존 LG 방식을 유지합니다.
LG/ALG의 선택 출력에 iBKD용 256채널·16×16 선행 adapter를 일괄 적용하지 않습니다.
Controller도 후속 사용자 지시 "1에폭마다 측정"에 따라 아래 정의로 고정했습니다.
β 선별 및 가중치·코드 검증은 남은 실행 작업이며 새로운 사용자 선택과 구분합니다.

### FSKD 공개 코드에서 확인된 구현 공백

확인한 저장소는 [TouchNow/FSKD](https://github.com/TouchNow/FSKD/tree/969dddf278b2c9f2dadde504326fa9d704c5a5aa),
commit `969dddf278b2c9f2dadde504326fa9d704c5a5aa`입니다.
[README](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/README.md)는
CIFAR/ImageNet 분류 실행 예시를 제공하며, 해당 tree에는 Cityscapes/SegFormer 학습 config가 없습니다.

2026-09-23에 GitHub API와 저장소 전체 clone으로 공개 범위를 추가 확인했습니다.
공개 branch는 `master` 하나이고, 전체 3개 commit의 Python·config·문서·shell 파일에서
Cityscapes/SegFormer/DeepLab/MMSegmentation 구현이나 실행 설정을 찾지 못했습니다.
최신 tree의 config는 `configs/cifar/`와 `configs/imagenet/` 아래 5개 YAML입니다.
별도 tag·release·공개 issue·PR은 조회 결과 모두 0개였습니다.
이는 확인한 저장소의 공개 상태에 대한 기록이며, 저자의 비공개 구현 유무를 판단한 것은 아닙니다.

논문 전문 확인 범위는 명세에 해시를 고정한 2025 SSRN preprint입니다.
[2026 저널 최종판](https://www.sciencedirect.com/science/article/pii/S0893608026000638)은
검색으로 초록과 일부 내용을 확인했으나, 본문 직접 접근이 제한되어 전문·부록 전체를
검증하지 못했습니다. 별도 공개 부록도 이번 검색에서는 찾지 못했으므로,
미확인 항목이 최종판이나 저자 제공 자료에도 없다고 단정하지 않습니다.

Cityscapes 완전 재현을 위해 필요한 추가 자료는 저자의 segmentation 학습 script/config,
정확한 teacher·student 초기 가중치, FSKD 각 loss의 계수·reduction·temperature,
feature pairing과 CLS token이 없는 MiT에서의 attention 처리, 증강·평가 설정입니다.
CIRKD에서 공통 조건을 보완하는 것만으로 FSKD 고유 구현의 공백까지 확인되는 것은 아닙니다.

기본값이 전혀 없는 것은 아닙니다. 공개
[train.py](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/train.py)와
README에서 확인되는 구조 손실 가중치는 다음과 같습니다.

| 적용 범위 | global | patch | attention | 해석 |
|---|---:|---:|---:|---|
| train.py 인자 기본값 | 1 | 1 | 1 | 파서 기본값; Cityscapes 실험값으로 확인한 것은 아님 |
| README CIFAR-100 / PiT-Ti 예시 | 1 | 1 | 40,000 | 분류 실행 예시의 명시값 |
| README ImageNet / DeiT-Ti 예시 | 100 | 1 | 1,000,000 | 다른 분류 실행 예시의 명시값 |
| Cityscapes / MiT-B0 | 미확인 | 미확인 | 미확인 | 확인한 프리프린트와 위 코드에서 전용 조합을 확정하지 못함 |

KD loss 계수와 temperature의 파서 기본값은 각각 1이며, 사용 stage 기본값은 1·2·3·4입니다.
README 예시는 stage도 각각 3·4와 1·2로 덮어씁니다. 따라서 파서 기본값 전체나 한 분류
명령을 Cityscapes의 저자 설정이라고 단정하지 않습니다. Cityscapes 전용 설정이 추가 확인되면
별도 revision으로 반영하며, 본 계획은 사용자에게 그 설정을 처음부터 탐색하도록 요구하지 않습니다.

[SimiKD 구현](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/distillers/simi.py)의
attention loss는 분류용 첫 토큰의 attention을 사용합니다. 반면 이번
[CIRKD MiT-B0](https://github.com/winycg/CIRKD/blob/48eb81b7a0ed7c59f9e347e97feb95a23e943d12/models/segformer.py)는
CLS token 없이 공간 토큰을 처리하고 stage별 spatial reduction을 사용합니다.
따라서 분류 코드를 그대로 연결한 것을 논문의 segmentation 구현이라고 부를 수 없습니다.

후속 요청으로 segmentation용 stage pairing, attention 집계, 정규화, KD temperature와
계수를 [별도 명세](BASELINE_METHOD_PROTOCOLS.md)에 선정했습니다. 공개 CIFAR/PiT 조합인
1/1/40,000과 stage 3·4를 함께 이식하고 B0의 CLS 없는 attention을 보완했습니다.
이를 **공통 조건에서의 FSKD 재구현**으로 표시하며 공식 Cityscapes 권장값이라고 주장하지 않습니다.
구현 후 각 항이 유한하며 student로 gradient를 실제 전달하는지도 확인합니다.

### Controller 확정 설정과 구현 확인

ALG의 guidance warm-up은 앞선 사용자 결정대로 **0**을 유지합니다.
기존 2k 실행은 **iBKD도 guidance warm-up 0 epoch**였습니다.
2026-09-24 사용자 결정에 따라 새 revision의 **iBKD lambda 0.25·0.5만 20 epoch**로 변경합니다.
공통 LR warm-up 0과 guidance 종료를 막는 warm-up은 서로 다른 설정입니다.
두 controller의 window **50**과
threshold **-0.02**는 기존 L/16 설정을 유지합니다.

이번 데이터 loader는 iteration 기준의 확장 목록을 사용하므로 loader 전체 순회가
통상적인 1 epoch가 아닙니다. 사용자 지시의 1 epoch 관측은 앞서 설명한
**`ceil(2975/16)=186` optimizer step 분량마다 관측**으로 고정합니다.
이는 2,976개 sample presentation 구간이며 고유 train 이미지가 한 번씩 등장한다는
뜻은 아닙니다. 이 경계에서 sampler를 다시 섞거나 초기화하지 않습니다.
기존 warm-up 0에서는 첫 관측에 손실 변화량이 없으므로 **step372 끝**에서 가장 빠른 종료 판정,
**step373부터** guidance off가 가능합니다. 이는 ALG에 계속 적용됩니다.
새 iBKD warm-up 20은 손실을 처음부터 관측하되 20번째 관측인 **step3720 끝**부터 판단하며,
**step3721부터** guidance off가 가능합니다. 종료 조건을 만족하지 않으면 계속 사용합니다.

관측값은 해당 구간의 **beta를 곱하기 전 raw guidance의 표본 수 가중 평균**입니다.
ALG는 locality loss, iBKD는 `(1-lambda)*alignment + lambda*fusion`을 관측합니다.
기존 [GuidanceController](../../src/ibkd_seg/phase1/controllers.py)의 초기 구간 계산과
평활화 수식을 유지하며 ALG는 `>= -0.02`, iBKD는 `> -0.02`일 때 종료합니다.
종료가 결정된 구간까지는 기존 beta로 학습하고, 바로 다음 optimizer step부터 beta를 0으로
만듭니다. LG는 같은 주기로 평균 guidance를 기록하되 자동 종료하지 않습니다.

500/2k/10k 중단점이나 checkpoint 저장 때문에 불완전한 구간을 추가 관측하지 않습니다.
완료 관측 수, 현재 구간의 step 수·손실 가중합·표본 수, beta와 controller 전체 상태를
저장하고 재개 후 같은 구간을 이어 계산합니다. 최종 80,000 step의 마지막 불완전 구간도
진단 loss는 기록하되 controller에 추가 epoch로 전달하지 않습니다.
Val 평가는 이 관측 주기와 별개로 공통 프로토콜의 **400 step 간격**을 유지합니다.

Controller의 강제 경계·off·resume 검사와 실제 학습에서의 자연스러운 off 관측을 구분합니다.
LG/ALG는 controller가 켜져 있는 동안 같은 β·초기화·입력에서 동일한 손실과 update가
나오는지 짧게 확인합니다. iBKD는 같은 설정의 반복·재개를 확인하며 LG와 같아야 하는 것은
아닙니다. 이전에 검증한 연산을 모두 장시간 재검사하기보다 B0에서 달라진 연결·상태를 확인합니다.

## 3. 초기 손실 측정과 후보 구성

먼저 B0의 공통 초기 상태와 seed1 입력을 고정합니다. 아래 측정 절차는 실행을 완료했고
[결과와 파생 grid](reports/h200_beta_calibration_v1/RESULTS.md)를 보존했습니다.

- 공통 초기 student를 복원한 상태에서 같은 25개 학습 batch를 이용해 CE와 방법별 raw
  guidance를 측정합니다. 이 단계에서는 optimizer update를 하지 않으며, 측정 중 변한
  BN/RNG 상태까지 복원합니다. 연결 smoke는 v2에서 완료했으며 후보 학습은 별도 실행합니다.
- LG/ALG는 동일한 guidance와 공통 β grid를 사용합니다.
- iBKD는 alignment와 fusion을 각각 기록하고 λ별 합성 guidance를 계산합니다.
- 모든 항이 유한한지, mask/축/정규화가 올바른지, student와 adapter에 필요한 gradient가
  전달되는지 먼저 확인합니다. 거의 0인 guidance에 β를 무작정 크게 주지 않습니다.

LG/ALG/iBKD의 초기 탐색 범위는 다음과 같이 구성할 수 있습니다.

```text
C = 고정된 batch들에서 측정한 CE의 중앙값
G = 같은 batch들에서 측정한 해당 방법의 raw guidance 중앙값
r = [0.03, 0.07, 0.15, 0.30]
β 후보 = r × C / G
```

이는 guidance가 초기 CE의 대략 3%, 7%, 15%, 30%가 되는 범위를 잡는
**탐색용 경험 규칙**입니다. 논문이 지정한 β나 최적성 보장이 아닙니다.
중앙값의 비로 계산하므로 실제 batch별 `βG/CE`의 중앙값·범위도 다시 기록합니다.
숫자는 사전 명세대로 유효숫자 6자리로 반올림하되 네 후보가 겹치지 않도록 하고,
수치·계산 근거·실측 비율을 validation 성능을 보기 전에 고정합니다.

CE와 guidance의 숫자 비율이 같아도 gradient의 크기·방향이 같지는 않습니다.
고정한 일부 batch의 student CE/guidance gradient norm은 H200 smoke v2에서 확인했습니다.
이번 25-batch calibration은 `torch.no_grad()`만 사용하며 backward를 수행하지 않습니다.
학습 중에는 CE, raw guidance, weighted guidance, 실제 β, gradient norm을 기록합니다.
전 step 전체 gradient 해시는 본학습의 필수 조건이 아닙니다.

FSKD는 별도 방법 명세의 고정된 가중치·temperature·feature 연결을 사용합니다.
Global/patch/attention/logit 항별 손실과 gradient를 확인하는
smoke를 수행합니다. 이 점검은 구현 확인이며 최적 가중치를 다시 찾는 sweep이 아닙니다.
저자 설정을 끝내 확보하지 못한 항목은 미확인으로 남기고, 재구현을 위해 선택하는 값의
근거를 별도 기록합니다. 다른 방법의 β 탐색에 맞춰 FSKD에도 임의로 4개 조합을 요구하거나,
분류 기본값을 Cityscapes 확정값으로 표시하지 않습니다. 추가 튜닝은 별도 실험으로 취급하고
저자 설정 재현과 구분합니다. 각 방법의 설정 출처와 실제 탐색 비용을 결과에 함께 기록합니다.

## 4. 짧은 실행부터 최종 비교까지

아래 표는 완료한 **warm-up 0의 최초 선별 계획**을 보존합니다. 최신 iBKD 계획은 λ별 β4개를
초기화부터 10k까지 실행해 1개를 선택하는 것으로 변경했습니다. 2k는 경과 기록이며
기존 iBKD 2k checkpoint는 승계하지 않습니다. 변경 없는 방법은 아래 최초 선별 규칙을 유지합니다.

모든 단계는 seed1, 동일한 데이터·student 초기화와 **80,000-step LR schedule**을
사용합니다. 500/2k/10k는 관찰·중단 지점입니다. Schedule 분모를 500/2k/10k로 줄이지 않습니다.

| 단계 | 실행 범위 | 판단과 다음 단계 |
|---|---|---|
| 연결 smoke | 각 방법 25~100 step 수준 | loss/gradient·실제 batch·입력·재개·controller 경계 확인; β 확정 전의 진단 실행 |
| 500-step 확인 | 아래 2k 후보 실행의 첫 500 step | NaN/Inf 등 실행 불능 확인; 유한한 일시적 loss 상승만으로 영구 제외하지 않음 |
| 2,000-step 선별 | β 탐색 네 조건 × 4후보 + 고정 FSKD·Vanilla | β 조건별 상위 2개 유지; FSKD·Vanilla는 중간 점검 |
| 10,000-step 선별 | β 조건별 상위 2개 + 고정 FSKD·Vanilla | β 조건별 최종 1개 선택; baseline은 같은 설정으로 계속 |
| 80,000-step seed1 | 선택된 β 조건 4개 + 고정 FSKD·Vanilla | 6개 조건의 최종 성능 비교 |
| 추가 seed | 선택 설정을 바꾸지 않고 seed2·3 | 최종 조건별 반복, 평균·표준편차 보고 |

FSKD 방법 명세가 준비된 경우 후보 수는 다음과 같습니다. 모든 후보가 정상 진행한다고
가정한 **서로 다른 실행 궤적 수**이며, smoke·재개 작업 횟수는 포함하지 않습니다.

| 조건 | 2k 후보 | 10k 후보 | 80k 최종 |
|---|---:|---:|---:|
| FSKD* | 명세의 재구현 고정값 1개 | 같은 설정 계속 | 같은 설정 계속 |
| LG | β 4개 | 2개 | 1개 |
| ALG | LG와 공통 β 4개 | 2개 | 1개 |
| iBKD λ=0.25 | β 4개 | 2개 | 1개 |
| iBKD λ=0.5 | β 4개 | 2개 | 1개 |
| Vanilla | 1개 | 같은 실행 계속 | 같은 실행 계속 |
| 합계 | 18개 | 10개 | 6개 |

500 step을 마친 같은 실행을 2k까지 이어갑니다. 안전하게 재개 가능한 후보만 2k→10k,
10k→80k로 이어가므로 이전 구간을 반복할 필요가 없습니다. 단, 선별은 최종 순위를 보장하지
않습니다. 2k는 초반 학습 편향이 크며, warm-up 0인 iBKD가 이미 종료됐을 수도 있으므로
실제 on/off와 stop step을 함께 기록합니다. 2k만으로 장기 성능을 확정하지 않습니다.
이번 계획은 아래 동점 처리 규칙을 포함한 상위 2개 선별을 사용합니다.
추가 후보 검증이 필요하면 별도 탐색으로 기록하고 최초 비교와 누적 탐색 비용을 구분합니다.
FSKD와 Vanilla에는 후보 순위 선별을 적용하지 않습니다. 초기 mIoU가 낮다는 이유만으로
최종 비교에서 제외하지 않고, 고정한 설정으로 80k까지 진행합니다. 실행 오류는 먼저 진단합니다.

### 선택·중단 규칙

- 정규 평가는 공통 규칙대로 매 400 step 전체 val 500장입니다. 2k에는 5회, 10k에는
  25회, 80k에는 200회의 정규 평가가 포함됩니다. 500-step 중간 확인에는 step400 평가가
  있으며 별도의 축소 val 수치를 후보 순위에 섞지 않습니다.
- β를 탐색하는 네 조건은 각 예산까지의 **best val mIoU**로 후보를 정렬합니다. 동점이면 더 이른 best step,
  다시 동점이면 사전에 고정한 candidate ID 순서로 결정합니다. Last mIoU와 학습 곡선도
  함께 남겨, 우연한 한 번의 개선인지 검토할 수 있게 합니다.
- NaN/Inf loss·gradient·parameter는 해당 실행을 중단하고 마지막 정상 상태를 보존합니다.
  원인이 모델 공통 구현인지 특정 후보인지 진단합니다. 실패 실행과 재실행을 모두 기록하며
  다른 β로 몰래 바꿔 같은 run ID에 덮어쓰지 않습니다.
- OOM·다운로드 실패·입력 불일치는 성능 열세가 아니라 실행 실패로 분류합니다.
  유한한 CE 급등이나 초반 낮은 mIoU만으로 회복 불가능한 발산이라고 단정하지 않습니다.
- 유효 후보가 2개 미만이면 남은 실패 후보를 억지로 통과시키지 않습니다. 원인 점검 후
  새 grid revision으로 범위를 조정하고 변경 이유·추가 탐색 비용을 기록합니다.
- Test는 후보 선정·checkpoint 선택에 쓰지 않습니다. λ=0.25와 0.5는 결과표에 각각
  남기며 둘 중 잘 나온 것만 골라 하나의 iBKD 결과인 것처럼 보고하지 않습니다.

### 재개 조건과 시간 예산

재개는 student/adapter/optimizer/RNG/sampler/controller, completed step과 best metadata가
모두 보존되고, protocol·method·candidate·asset·코드 버전이 일치할 때만 허용합니다.
짧은 run에서 뽑은 best 가중치가 아니라 **정확한 중단 step의 전체 상태**에서 이어갑니다.
후보를 선별한 뒤 β·loss를 바꾸거나 optimizer를 초기화하면 새로운 실험입니다.

기존 L/16의 2k·10k checkpoint는 이 B0 재개 규칙의 대상이 아닙니다.
FSKD/B0의 짧은 H200 smoke 학습 속도는 측정했습니다. 전체 val500과 저장 비용은 새
2k runner에서 따로 측정하고, 전체 시간은 학습과 매 400-step 평가 비용을 합산해 갱신합니다.
작업당 10시간 제한에는 설치·검증·저장 여유를 두며 최대 9시간 실행 후 전체 상태를 저장합니다.
1회 validation과 저장에 필요한 시간을 실측해 종료 전에 충분한 여유를 남깁니다.

## 5. 결과 기록과 현재 남은 작업

각 실행의 최종 로그 JSON에는 다음을 포함하도록 B0 runner를 구현합니다.

- protocol/method/candidate ID, seed, β, iBKD λ 또는 FSKD 가중치 조합, LR schedule horizon
- 완료 step·목표 step·상태, 종료 이유, runtime·평가·저장 시간과 peak GPU memory
- 마지막 CE·각 guidance·weighted guidance·전체 loss, 실제 guidance on/off와 stop step
- best step·controller 관측 구간, 같은 best checkpoint의 val mIoU·pixel accuracy·19 class IoU,
  last-step 지표와 metric scale(%)
- teacher/student/초기화/입력 식별 해시, 반복·재개 확인 결과, checkpoint 위치, `test_used=false`

평가하지 않은 지표는 null과 그 이유를 기록합니다. 2k/10k 결과는 후보 선택용임을 표시하고
80k 결과와 같은 최종 표에 섞지 않습니다. FSKD 저자 보고값은 별도 참고값으로 구분합니다.

현재 완료된 것은 **공통 조건 고정, 비교 범위와 위 흐름 정리, iBKD 전체 8개 block의
256채널·16×16 가중합 규격, LG/ALG의 1·5·8번째 block 선택, 186-step controller 관측과
기존 window/threshold 유지, 최초 ALG·iBKD guidance warm-up 0**입니다.
후속 결정으로 iBKD만 warm-up 20인 별도 실행 명세·진입점·경계 재개 검사를 준비했습니다.
ALG는 0을 유지합니다. 새 iBKD의 H200 실험은 아직 시작하지 않았습니다.
FSKD·C2VKD 방법별 재구현, 실제 가중치 식별, 7개 방법 H200 smoke와 25-batch β 측정을
완료했습니다. 2k runner의 전체 상태 재개·전체 val·후보 선택 구현 및 로컬 검사도 완료했습니다.
사용자 지정 묶음은 Vanilla·FSKD·LG / ALG·iBKD λ=0.25 / iBKD λ=0.5입니다.
C2VKD는 CLIP 대체 pool을 가진 별도 후보로, 이 세 이슈에는 포함하지 않습니다.
NVIDIA HF B0 초기화 출처는 2k 실행 명세에 명시적으로 승계하며 원래 공통 JSON은 보존합니다.
