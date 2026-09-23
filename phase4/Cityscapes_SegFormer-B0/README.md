# Cityscapes · SegFormer-B0 비교 실험

2026-09-23 H200 v1에서 **7개 방법 모두 실제 입력의 3 update를 완료했으나 checkpoint 재실행 비교에 실패**했습니다.
[로그 점검](reports/h200_smoke_v1_819/RESULTS.md)에 근거와 제한을 기록했습니다.
결정적 연산 설정·복원 진단·실패 결과 보존을 보완한 **v2를 준비**했습니다.
v2는 로컬 단위 검사 14개 및 CPU 연결·재실행 7/7을 통과했고 GPU 재검사가 남았습니다.
β 선별·80k 본실험 runner도 아직 실행하지 않았습니다.
바로 실행할 입력값은 [H200 smoke 이슈](H200_SMOKE_ISSUE.md)에 있습니다.
이번 smoke는 NVIDIA ImageNet MiT-B0 배포본을 역변환해 사용하며,
CIRKD Baidu 파일과의 동일성은 미확인입니다. 이 출처 예외는 본실험과 구분합니다.

목적은 같은 **DeepLabV3-R101 teacher → SegFormer MiT-B0 student**에서
FSKD, LG, ALG, iBKD λ=0.25·0.5를 비교하는 것입니다.
증류 없이 학습했을 때의 성능을 확인하기 위해 Vanilla도 포함합니다.

## 먼저 읽을 파일

| 파일 | 내용 |
|---|---|
| [H200 smoke 이슈](H200_SMOKE_ISSUE.md) | 7개 방법의 실행 명령·검사 항목·성공 판정 |
| [로컬 검사 기록](SMOKE_PREPARATION.md) | 실제 가중치 CPU 연결·재개 검사와 남은 GPU 검증 |
| [PROTOCOL.md](PROTOCOL.md) | 공통 학습·평가 조건, 각 값의 출처와 선택 이유 |
| [공통 프로토콜 JSON](configs/segformer_b0_common_protocol_v1.json) | 고정값과 출처 버전을 담은 명세. 실행 config 아님 |
| [비교 범위 JSON](configs/comparison_manifest_v1.json) | 공통 JSON 해시, 요청한 다섯 KD 조건과 Vanilla, 선별 규칙·남은 항목 |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 방법별 준비, β 후보 구성, 2k·10k·80k 진행과 선택 규칙 |
| [공식 baseline 자료 점검](OFFICIAL_BASELINE_AUDIT.md) | FSKD·C2VKD 공식 저장소의 공개 범위, 누락 코드·설정, 비교표 적용 판단 |
| [FSKD·C2VKD 방법 프로토콜](BASELINE_METHOD_PROTOCOLS.md) | 공개값·이식값·보완값을 구분한 연결, 손실, 계수 및 C2VKD pooling 대체안 |
| [FSKD JSON](configs/fskd_method_protocol_v1.json) / [C2VKD JSON](configs/c2vkd_method_protocol_v1.json) | 첫 재구현의 수치 명세. 실행 config 아님 |

공통 조건은 **FSKD 명시값 → CIRKD 2022의 SegFormer 코드 → NVlabs 공식 B0**
순서로 채택했습니다. fine train 2,975장 / val 500장, crop 512×512, 총 batch 16,
AdamW, LR 6e-5, poly power 0.9, 80,000 step을 사용합니다.
매 400 step 전체 val을 평가하고 **val mIoU**로 checkpoint를 선택합니다.
출처의 누락값을 보완한 공통 비교 조건이며, FSKD 저자의 전체 실험을 그대로 재현했다는
뜻은 아닙니다.

## 비교 조건

| 조건 | 고정하거나 선별할 내용 | 현재 상태 |
|---|---|---|
| Vanilla | 공통 CE만 사용, teacher 없음 | GPU 3 update 확인; 재개 v2 재검사 전 |
| FSKD* | stage 3·4, global/patch/attention=1/1/40,000, KD T=1 | 재구현 GPU 3 update 확인; 재개 v2 재검사 전 |
| LG | 처음·중간·끝인 1·5·8번째 block, β 후보 선별 | GPU 3 update 확인; 재개 v2·β 선별 전 |
| ALG | LG와 같은 연결, 1 epoch 분량마다 종료 판단, guidance warm-up 0 | GPU 3 update 확인; 재개 v2·β 선별 전 |
| iBKD λ=0.25 | 전체 8개 block을 256채널·16×16로 맞춰 가중합, 1 epoch 분량마다 종료 판단, β 선별 | GPU 3 update 확인; 재개 v2·β 선별 전 |
| iBKD λ=0.5 | 위와 동일한 연결·controller, 이 λ 조건에서 β 선별 | GPU 3 update 확인; 재개 v2·β 선별 전 |

추가 요청에 따라 **C2VKD 방법 프로토콜도 작성**했습니다. 공개된 세 feature loss와
논문 기반 PDD를 연결하며, 원본 attention-pooling 가중치가 없어 CLIP RN101 pool을
사용하는 `C2VKD* (CLIP-pool)` 대체안을 별도로 명세했습니다. 추가 사전학습 조건이
생기므로 기존 6개 주 비교에 자동 편입하지 않습니다. C2VKD는 원래 목적함수대로
PDD가 CE를 대신하는 방법별 예외도 명시했습니다.

**0.25·0.5는 iBKD 내부 손실 혼합비 λ입니다. β는 전체 guidance의 강도이며 별도로
찾습니다.** L/16에서 선정한 β는 B0에서 검증된 값이 아닙니다.
FSKD의 손실별 λ1·λ2·λ3도 iBKD의 λ와 서로 다른 파라미터입니다.

2026-09-23 후속 논의에서 iBKD의 가중합 직전 목표를 **256채널·16×16**으로
선택했습니다. 이는 첫 실험에 사용할 설정이며 최적성이나 GPU 검증이 확인된 값은 아닙니다.
기존 분류의 전체 block adapter를 기준으로 B0의 8개 block 전체를 연결합니다.
이 규격을 LG/ALG/FSKD에 일괄 적용하거나 fusion 전체의 계산 해상도로 해석하지 않습니다.
이어 사용자 결정으로 LG/ALG는 **1·5·8번째 block**(코드 인덱스 `[0,4,7]`),
controller는 **1 epoch 분량인 186 step마다 관측**하도록 고정했습니다.
고정된 확장 목록 loader에서 `ceil(2975/16)=186`으로 정의하며 고유 이미지가
한 번씩 등장하는 실제 데이터 순회와는 구분합니다. Window 50, threshold -0.02는 유지합니다.
두 controller의 guidance warm-up은 **ALG 0·iBKD 0 epoch**입니다.
iBKD도 warm-up 없이 시작하라는 후속 사용자 결정을 두 lambda 조건에 동일하게 적용합니다.
Controller 관측과 별개로 val 평가는 공통 조건인 400 step마다 수행합니다.
확정 범위와 남은 명세는 [연결 상태 점검](EXPERIMENT_PLAN.md#2026-09-23-후속-연결-상태-점검)에 기록했습니다.

공통 조건과 비교 범위의 고정 결과는 위 두 JSON을 한 쌍으로 관리합니다.
비교 범위 JSON은 공통 JSON의 SHA-256을 참조하며, 실제 구현·검증이 남은 항목을
`null`과 상태값으로 구분합니다. FSKD의 **저자 Cityscapes 확인값**은 미확인으로 남기되,
이번에 선정한 **우리의 재구현값**은 별도 방법 JSON에 기록했습니다. 수치 β는 아직 미측정입니다.

## 실행 순서

1. 구현된 B0 smoke를 H200에서 실행해 실제 입력의 손실·gradient·재개·메모리를 확인합니다.
2. smoke 통과 후 전체 runner를 완성하고 계획의 25-batch 측정으로 β 후보 범위를 정합니다.
   smoke의 3-batch 임시 β는 최종 선별 후보가 아닙니다.
3. LG, ALG, iBKD 두 λ 조건은 각각 β 4개를 구성해 2,000 step까지 선별합니다.
   첫 500 step은 같은 실행 안에서 안정성을 확인하는 구간입니다.
4. β를 탐색하는 네 조건의 상위 2개를 10,000 step까지 비교합니다.
   FSKD는 위 명세의 재구현 고정값으로 smoke·중간 점검을 진행하며,
   반드시 가중치 후보를 새로 탐색해야 하는 것은 아닙니다.
5. 네 조건에서 각각 β 1개를 선택해, 고정한 FSKD 및 Vanilla와 80,000 step까지
   최종 **6개 조건**을 비교합니다. 이후 설정을 고정한 채 추가 seed로 반복합니다.

짧은 선별도 **80k LR schedule의 앞부분**으로 실행합니다. 동일 설정·전체 상태를
보존했다면 선별 checkpoint에서 이어갈 수 있고, 모델·loss·β·schedule을 바꾸었다면
초기화부터 다시 시작해야 합니다. 구체적인 기준은 [실험 흐름](EXPERIMENT_PLAN.md)에 있습니다.

본실험 β 후보는 **미선별**입니다. H200 v1의 3-update loss와 실패 원인 검토는 기록했으며,
다음 작업은 [H200 smoke v2](H200_SMOKE_ISSUE.md)의 재개·평가 재검사입니다.
FSKD의 저자 비공개 설정을 확인한 것은 아니며, C2VKD 대체안은 별도 후보로 기록합니다.

## 기존 실험과의 관계

- [Segmenter-L/16 기록](../phase4_cityscapes/README.md): 기존 2k·10k 결과와 80k 후속 안내.
- [L/16 10k 결과](../phase4_cityscapes/reports/candidate_selection/l16_crop512_candidate_top2_grid10000_v18/RESULTS.md): 과거 후보 선택 근거.
- 이 폴더는 B0 전용입니다. L/16의 checkpoint·β·실행 시간을 B0 결과로 이전하지 않습니다.
- 새 JSON은 기존 L/16 config 위치에서 **내용 변경 없이 이동**했습니다.
  프로토콜 ID는 `cityscapes_deeplabv3r101_segformerb0_common_v1`로 유지합니다.
- 데이터셋, 가중치, 원시 결과는 Git 밖에 보관하고 검증 해시와 정리된 결과만 기록합니다.
