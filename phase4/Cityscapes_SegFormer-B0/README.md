# Cityscapes · SegFormer-B0 비교 실험

**2026-09-24 최신 결정: iBKD λ=0.25·0.5만 guidance warm-up 20으로 재실험하고 ALG는 0을 유지합니다.**
**지금 실행할 순서:** [λ=0.25의 β 4개 smoke](H200_WARMUP20_SMOKE_ISSUE.md)를 먼저 확인한 뒤,
같은 네 β를 각각 2,000 step까지 실행하고 결과를 확인합니다. 10k를 바로 실행하지 않습니다.
[변경 프로토콜](WARMUP20_PROTOCOL.md)과 [H200 실행 이슈 2개](H200_WARMUP20_ISSUES.md)를 준비했습니다.
첫 3720 step의 guidance를 보장하며, 각 λ의 기존 β 4개를 초기화부터 10k까지 비교합니다.
새 iBKD의 2k는 경과 기록입니다. 아래 기존 warm-up 0 결과·선정 기록은 보존하며 새 실행에 재개하지 않습니다.

2026-09-23 **H200 smoke v2에서 7개 방법 모두 학습·checkpoint 재개·val2 평가를 통과**했습니다.
[최신 결과](reports/h200_smoke_v2/RESULTS.md)에 방법별 검사·메모리와 확인 범위를 기록했습니다.
v1의 재실행 불일치는 v2의 고정 기준에서 해소됐습니다.
이어 [H200 25-batch calibration](reports/h200_beta_calibration_v1/RESULTS.md)도 통과해 β 후보를 고정했습니다.
기존 실행 입력값은 [2,000-step 선별 이슈 3개](H200_SCREEN2000_ISSUES.md)에 보존했습니다.
첫 H200 2k pack1은 6개 실행 모두 25 update 후 공통 데이터 로더 오류로 실패했습니다.
[실패 점검](reports/h200_screen2000_v1_pack1_failure/RESULTS.md)에 원인과 마지막 손실을 기록했습니다.
2026-09-24 수정한 **2k v2 pack1의 6개 실행이 모두 완료**됐습니다.
[pack1 결과](reports/h200_screen2000_v2_pack1/RESULTS.md)에서 전체 val500, 입력·재개 검사와
평가 집계를 확인했고 LG는 **β=0.197479·0.460784**를 10k 후보로 유지합니다.
Vanilla best mIoU 45.29%, FSKD 35.07%, LG 최고 44.76%이며 최종 성능 결론은 아닙니다.
이어 [pack2 후보별 결과](reports/h200_screen2000_v2_pack2/RESULTS.md)에서도 8개 후보의 2k 완료를 확인했습니다.
첨부 앞부분이 잘려 전체 group 집계는 미확인이나, 후보별 재계산 결과 ALG는
**β=0.197479·0.460784**를 유지합니다. 당시 iBKD λ=0.25는 **β=0.903048·3.87021**을 선정했습니다.
[pack3 결과](reports/h200_screen2000_v2_pack3/RESULTS.md)도 4/4 완료했습니다.
iBKD λ=0.5는 최고 mIoU 45.6926%이며 당시 **β=0.4303·1.00403**을 선정했습니다.
기존 계획은 10k 대상 10개였으나, 새 계획은 변경 없는 6개 재개와 iBKD warm-up 20의
8개 신규 실행을 합한 **14개 실행 궤적**입니다. iBKD의 기존 상위 2개 제한은 새 revision에 적용하지 않습니다.
10k·80k 결과는 아직 없습니다. Pack2의 누락된 원본 집계는 재개 준비 때 확인해야 합니다.
이번 선별도 NVIDIA ImageNet MiT-B0 배포본 역변환을 사용합니다.
CIRKD Baidu 파일과의 동일성은 미확인이며, 새 실행 명세에 이 출처를 명시했습니다.

목적은 같은 **DeepLabV3-R101 teacher → SegFormer MiT-B0 student**에서
FSKD, LG, ALG, iBKD λ=0.25·0.5를 비교하는 것입니다.
증류 없이 학습했을 때의 성능을 확인하기 위해 Vanilla도 포함합니다.

## 먼저 읽을 파일

| 파일 | 내용 |
|---|---|
| [지금 실행할 warm-up 20 smoke](H200_WARMUP20_SMOKE_ISSUE.md) | λ=0.25의 β 4개 × 32 step·val2, 통과 후 별도 β4개·2k |
| [iBKD warm-up 20 프로토콜](WARMUP20_PROTOCOL.md) | 최신 사용자 결정, 정확한 종료 경계·기존 결과와 분리·10k 재선별 |
| [iBKD warm-up 20 이슈 2개](H200_WARMUP20_ISSUES.md) | λ별 β 4개를 초기화부터 10k 실행, 전체 입력값과 재개 |
| [H200 2k 이슈 3개](H200_SCREEN2000_ISSUES.md) | 사용자 지정 6개·8개·4개 실행, 내장 재개 검사와 전체 val500 |
| [H200 β 후보 생성 이슈](H200_BETA_CALIBRATION_ISSUE.md) | 25-batch 초기 손실 측정·β 4개씩 생성, 학습·평가 없음 |
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
| Vanilla | 공통 CE만 사용, teacher 없음 | 2k v2 완료; best mIoU 45.29% |
| FSKD* | stage 3·4, global/patch/attention=1/1/40,000, KD T=1 | 2k v2 완료; best 35.07%, last 32.73% |
| LG | 처음·중간·끝인 1·5·8번째 block, β 후보 선별 | 2k 완료; β=0.197479·0.460784를 10k에 유지 |
| ALG | LG와 같은 연결, 1 epoch 분량마다 종료 판단, guidance warm-up 0 | 2k 후보 기록 확인; β=0.197479·0.460784를 10k에 유지 |
| iBKD λ=0.25 | 전체 8개 block을 256채널·16×16로 맞춰 가중합, 186-step 관측, guidance warm-up 20 | 기존 warm-up 0 결과 보존; 새 β 4개·10k 재선별 준비 |
| iBKD λ=0.5 | 위와 동일한 연결·controller, 이 λ 조건에서 β 선별 | 기존 warm-up 0 결과 보존; 새 β 4개·10k 재선별 준비 |

추가 요청에 따라 **C2VKD 방법 프로토콜도 작성**했습니다. 공개된 세 feature loss와
논문 기반 PDD를 연결하며, 원본 attention-pooling 가중치가 없어 CLIP RN101 pool을
사용하는 `C2VKD* (CLIP-pool)` 대체안을 별도로 명세했습니다. 추가 사전학습 조건이
생기므로 기존 6개 주 비교에 자동 편입하지 않습니다. C2VKD는 원래 목적함수대로
PDD가 CE를 대신하는 방법별 예외도 명시했습니다.

**0.25·0.5는 iBKD 내부 손실 혼합비 λ입니다. β는 전체 guidance의 강도이며 별도로
찾습니다.** L/16에서 선정한 β는 B0에서 검증된 값이 아닙니다.
FSKD의 손실별 λ1·λ2·λ3도 iBKD의 λ와 서로 다른 파라미터입니다.

2026-09-23 후속 논의에서 iBKD의 가중합 직전 목표를 **256채널·16×16**으로
선택했습니다. GPU 연결 smoke는 통과했으며 최적성이나 장기 학습 성능을 확인한 값은 아닙니다.
기존 분류의 전체 block adapter를 기준으로 B0의 8개 block 전체를 연결합니다.
이 규격을 LG/ALG/FSKD에 일괄 적용하거나 fusion 전체의 계산 해상도로 해석하지 않습니다.
이어 사용자 결정으로 LG/ALG는 **1·5·8번째 block**(코드 인덱스 `[0,4,7]`),
controller는 **1 epoch 분량인 186 step마다 관측**하도록 고정했습니다.
고정된 확장 목록 loader에서 `ceil(2975/16)=186`으로 정의하며 고유 이미지가
한 번씩 등장하는 실제 데이터 순회와는 구분합니다. Window 50, threshold -0.02는 유지합니다.
기존 2k 실행의 guidance warm-up은 **ALG 0·iBKD 0 epoch**였습니다.
2026-09-24 새 사용자 결정으로 **ALG 0 유지·iBKD 두 λ만 20 epoch**로 변경했습니다.
기존 고정 JSON은 보존하고 [별도 revision](configs/b0_ibkd_warmup20_v1.json)에 변경을 명시했습니다.
Controller 관측과 별개로 val 평가는 공통 조건인 400 step마다 수행합니다.
확정 범위와 남은 명세는 [연결 상태 점검](EXPERIMENT_PLAN.md#2026-09-23-후속-연결-상태-점검)에 기록했습니다.

공통 조건과 비교 범위의 고정 결과는 위 두 JSON을 한 쌍으로 관리합니다.
비교 범위 JSON은 공통 JSON의 SHA-256을 참조하며, 실제 구현·검증이 남은 항목을
`null`과 상태값으로 구분합니다. FSKD의 **저자 Cityscapes 확인값**은 미확인으로 남기되,
이번에 선정한 **우리의 재구현값**은 별도 방법 JSON에 기록했습니다.
수치 β 후보는 [25-batch 실측 grid](configs/b0_beta_grid_frozen_v1.json)에 고정했습니다.
LG·ALG는 기존 상위 2개를 유지하며, 새 iBKD는 λ별 β 4개를 10k까지 비교합니다.
최종 β는 각 조건의 10k 결과로 선정합니다.

## 실행 순서

1. H200 smoke v2에서 실제 입력의 손실·gradient·재개·val2·메모리 확인을 완료했습니다.
2. H200 25-batch 측정으로 β 후보 범위를 고정했습니다.
   smoke의 3-batch 임시 β는 최종 선별 후보가 아닙니다.
3. 기존 warm-up 0의 2k 선별은 완료했습니다. LG·ALG 상위 2개 및 고정 Vanilla·FSKD는
   동일한 전체 상태에서 10k로 이어갑니다.
4. 새 iBKD warm-up 20은 λ별 β 4개를 초기화부터 10k까지 실행합니다.
   2k는 중간 관측이며 10k에서 λ별 1개를 고릅니다. 첫 500-step 안정성 검사는 내장합니다.
5. 네 조건에서 각각 β 1개를 선택해, 고정한 FSKD 및 Vanilla와 80,000 step까지
   최종 **6개 조건**을 비교합니다. 이후 설정을 고정한 채 추가 seed로 반복합니다.

짧은 선별도 **80k LR schedule의 앞부분**으로 실행합니다. 동일 설정·전체 상태를
보존했다면 선별 checkpoint에서 이어갈 수 있고, 모델·loss·β·schedule을 바꾸었다면
초기화부터 다시 시작해야 합니다. 구체적인 기준은 [실험 흐름](EXPERIMENT_PLAN.md)에 있습니다.

다음 작업은 변경 없는 6개 실행의 10k 재개와 새 iBKD warm-up 20의 8개 후보 비교입니다.
동일한 2k 실행의 단순 반복이 아니라 iBKD guidance schedule을 변경한 별도 재실험입니다.
FSKD의 저자 비공개 설정을 확인한 것은 아니며, C2VKD 대체안은 별도 후보로 기록합니다.

## 기존 실험과의 관계

- [Segmenter-L/16 기록](../phase4_cityscapes/README.md): 기존 2k·10k 결과와 80k 후속 안내.
- [L/16 10k 결과](../phase4_cityscapes/reports/candidate_selection/l16_crop512_candidate_top2_grid10000_v18/RESULTS.md): 과거 후보 선택 근거.
- 이 폴더는 B0 전용입니다. L/16의 checkpoint·β·실행 시간을 B0 결과로 이전하지 않습니다.
- 새 JSON은 기존 L/16 config 위치에서 **내용 변경 없이 이동**했습니다.
  프로토콜 ID는 `cityscapes_deeplabv3r101_segformerb0_common_v1`로 유지합니다.
- 데이터셋, 가중치, 원시 결과는 Git 밖에 보관하고 검증 해시와 정리된 결과만 기록합니다.
