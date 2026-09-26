# Cityscapes · Segmenter-S/16

2026-09-26: **Tiny를 10,000step까지 먼저 진행한 뒤 Small을 시작**하기 위한 준비 폴더입니다.
H200 작업을 제출하거나 실행하지 않았습니다. Tiny의 결과·고정 β·기존 실행 설정은 변경하지 않습니다.
프로토콜은 [PROTOCOL.md](PROTOCOL.md), 첫 실행 설정은
[s16_smoke25_fskd_c2vkd_v1.json](configs/s16_smoke25_fskd_c2vkd_v1.json)에 있습니다.
이슈 입력용 MD 파일은 만들지 않습니다.

## 준비된 범위

- 공식 Small encoder 가중치의 다운로드·byte/SHA 검사, 원본 Segmenter factory 로딩.
- OpenMMLab DeepLabV3-R101-D8 teacher, Crop512·batch8·decoder1·seed1을 유지한 7개 경로.
  Vanilla, LG, ALG, iBKD λ=0.25, iBKD λ=0.5, FSKD*, C2VKD*입니다.
- 각 경로에서 초기 25batch 손실 측정(optimizer update 0회), 초기 상태 복원 후 25step 학습,
  마지막 update 저장/복원 재실행, val2 진단 평가.
- LG/ALG 및 두 iBKD 조건의 **Small 전용 β 후보 8개 산출**. 아직 수치는 미정입니다.
  Tiny의 수치나 순위를 Small에 복사하지 않습니다.
- 성공·실패 모두 마지막 JSON에 7개 실행 상태와 측정값을 모읍니다.

**현재 실행 가능한 것은 첫 25-step 스모크·calibration입니다.**
500/2k/10k 실행 설정은 이 스모크 결과로 β를 고정한 뒤 추가합니다.
Tiny의 500/2k 실행기는 Tiny config를 검증하므로 Small config를 임의로 넣지 않습니다.
25step 통과를 500step 안정성이나 장기 재현성 통과로 간주하지 않습니다.

## 나중에 시작할 명령

이 준비 변경을 포함한 commit을 고정한 H200 checkout에서 다음을 실행합니다.

```bash
bash phase4/Cityscapes_Segmenter-S16/scripts/run_smoke25.sh
```

실행 중인 Tiny와 같은 GPU에서 동시에 실행하지 않습니다. 모델·의존성 준비를 포함해
7개 방법을 순차 실행하며 후속 500/2k/10k 학습을 자동 시작하지 않습니다.
공개 초기 가중치는 내려받지만 Cityscapes 원본은 이미 전달한 ZIP을 사용합니다.

| 용도 | 기본 위치 / 설정 변수 |
|---|---|
| 원본 ZIP 2개 | `/app/data/chaoyang` · `CITYSCAPES_ZIP_DIR` |
| 기존 추출 데이터 | `/app/scratch/cityscapes_l16_crop512_v3/cityscapes` · `CITYSCAPES_CROP512_DATA_DIR` |
| Small 전용 소스·가중치 캐시 | `/app/scratch/cityscapes_s16_v1/upstream` · `CITYSCAPES_S16_CACHE` |
| Small 결과 | `/app/output/cityscapes_s16_crop512_smoke25_v1/run_<UTC>_<PID>/` · `CITYSCAPES_S16_OUTPUT` |

직접 출력 경로를 지정할 때는 아직 존재하지 않는 새 폴더를 지정합니다.
데이터 준비기는 기존 파일 내용이 같으면 재사용하고 다르면 덮어쓰지 않고 실패합니다.
train/val만 준비하고 test를 평가하지 않습니다. Tiny checkpoint는 필요하지 않습니다.

## 실험 진행 순서

| 단계 | 구성 | 확인할 내용 / 다음 단계 |
|---|---|---|
| ① 초기 점검 | 7개 × 초기 25batch + 25step | 손실·gradient·공통 입력·teacher 고정·LG/ALG 궤적·단일 update 재실행. Small β 후보 산출 |
| ② 500step | LG 8 + iBKD λ0.25 8 + λ0.5 8 = 24개 | 유한 손실/gradient, 저장 상태, 큰 불안정성 확인. val2 점수로 순위 확정 금지 |
| ③ 2,000step | 생존 후보의 LG·ALG·iBKD λ0.25·λ0.5 각각 실행, 전부 생존하면 32개 | 고정 endpoint의 전체 val500에서 accuracy·mIoU·19 class IoU 비교 |
| ④ 10,000step | 각 방법/λ의 상위 β 2개를 잠정 선정하면 8개 | 더 긴 구간의 성능과 controller 상태 확인. 2k 순위가 최종 순위라는 보장은 없음 |
| ⑤ 80,000step | 추후 선정한 조건 + 같은 길이의 Vanilla/비교군 | 전체 학습 성능 비교, 필요시 추가 seed. 지금 자동 실행하지 않음 |

LG·ALG는 같은 β·초기값·입력에서 controller 종료 전 학습 연산이 같습니다.
현재 controller는 빨라도 2epoch 완료(744step) 후 종료를 결정하므로 500step은 LG 검사로
공유할 수 있습니다. 2k부터는 ALG를 별도 실행합니다.
iBKD는 λ=0.25와 0.5를 항상 별도 조건으로 유지하고 20epoch warm-up을 유지합니다.
2k는 이 warm-up 내부이므로 가이던스 종료 후 성능을 확인하는 실험이 아닙니다.

FSKD*·C2VKD*·Vanilla는 β 탐색 대상이 아닙니다. 이들의 장기 비교를 수행할 때도 공통
데이터·초기 student·학습 길이·평가 endpoint를 맞춥니다. FSKD*/C2VKD*의 계수를
Small 결과에 맞춰 자동 변경하지 않습니다.

NaN/Inf·checkpoint 오류와 단순히 초반 점수가 낮은 후보를 구분합니다.
500step에서 낮은 accuracy/mIoU만으로 회복 불가능하다고 판단하거나 제외하지 않습니다.
중단/제외 규칙은 다음 단계 실행 전에 고정하고 결과를 본 뒤 바꾸지 않습니다.
Small의 step 시간과 전체 val500 시간을 새로 측정한 뒤 **10시간 이내 묶음**을 정합니다.
Tiny에서 측정한 시간으로 Small의 완료 시간을 보장하지 않습니다.

## 결과 확인

마지막 **`[CITYSCAPES_S16_SMOKE_FINAL]`**에는 7개 방법의 상태, λ, β 후보 8개,
초기 CE/guidance, 실제 초기 비율, 최종 loss/CE/guidance·고정 loss 성분,
선택 step/epoch, accuracy·mIoU·19 class IoU, teacher/저장 복원 검사,
step 시간·GPU 메모리·실패 이유가 들어갑니다.
출력은 50,000 ASCII bytes 이내로 제한해 서버의 마지막 65,000자 로그에 들어가게 합니다.
중간 로그는 기존처럼 출력하며, 최종 JSON을 위해 중간 출력을 없애지 않습니다.
강제 종료로 출력 기회 자체가 없을 때까지 마지막 줄을 보장하지는 않습니다.

- `run.log`: 설치부터 종료까지 전체 로그.
- `terminal_summary.json`: 마지막 JSON.
- `artifacts/smoke_summary.json`: 종합 결과, 원래 정밀도와 상세 provenance.
- `artifacts/<run_id>/calibration.json`: 초기 batch별 CE/guidance, 입력 해시, β 산출.
- `artifacts/<run_id>/summary.json`: 성공한 방법의 step별 loss·검사·metric.
- `artifacts/<run_id>/warnings.json`, `traceback.txt`: 경고 또는 실패 진단.

**val2 수치는 연결 확인용이며 전체 val 점수나 β 순위가 아닙니다.**
스모크 checkpoint는 마지막 update 재실행 검사 후 삭제되며 후속 500step의 시작점이 아닙니다.
후속 후보 비교는 같은 초기 가중치에서 0step부터 시작할 계획입니다.

## 기존 Tiny를 이어갈 때

Small 지원을 추가하면서 공유 코드의 source hash가 달라집니다.
실행 중인 Tiny는 이미 고정한 checkout을 계속 사용합니다. Tiny 2k 작업을 중단 후 재개할 때도
**그 작업의 원래 commit `6856e344bc32433f183cf1f1939f6812175ea84a`**와 복원된 후보 폴더를 사용합니다.
Small 준비 commit에서 오래된 Tiny checkpoint를 억지로 재개하거나 hash 검사를 끄지 않습니다.
이후 Tiny 10k도 실제 실행에 쓰는 commit/config를 별도로 고정합니다.

## 로컬 검증과 남은 확인

Small NPZ 88,851,254 bytes와 SHA256을 실제 다운로드 파일에서 확인했습니다.
고정 upstream 코드에서 S/16을 로드해 CPU에서 `[1,19,512,512]` 출력,
64×64 입력의 역전파·12개 feature·6-head CLS attention·optimizer update를 확인했습니다.
decoder1 포함 student는 **24,593,934 parameters**입니다.
Small 검사9개와 기존 Tiny55개·Large/공통11개, 총 **로컬 검사75개 통과**를 확인했습니다.
Small LG/iBKD 두 λ의 feature 연결과 역전파, FSKD attention/adapter gradient,
C2VKD adapter 차원, LG/ALG β 공유, 실패 후 다음 방법 실행, 설치 실패 종료 JSON을 포함합니다.
이 CPU 검사는 실제 Cityscapes/H200의 7개 smoke 경로 통과를 대신하지 않습니다.
GPU 메모리·CUDA 결정성·step 시간·β 수치·전체 val 성능은 향후 실행 결과로 확인해야 합니다.
