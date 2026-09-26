# Cityscapes · Segmenter-Ti/16

2026-09-26 사용자 결정: Tiny를 먼저 검사하며 **기존 OpenMMLab teacher를 유지**합니다.
이 폴더는 L/16 결과를 변경하지 않는 별도 실험입니다. 현재 구현 범위는 초기 손실 측정과
25-step 연결 smoke, 500-step β 후보 검사, 전체 val 평가 시간 측정입니다.
초기 #834의 LG/ALG 궤적 문제는 후속 v4 반복 검사에서 확인했고,
사용자 제공 v6 종료 로그에서 **24개 모두 500step 완료·공통 조건·저장 상태 비교 통과**를 확인했습니다.
Tiny의 2k/10k 결과는 아직 없습니다. 500step 점수는 val2 진단이므로 후보 순위 선정에 쓰지 않습니다.
이슈는 사용자가 제출하며 이슈 입력용 MD 파일이나 GitHub 이슈는 생성하지 않습니다.

**현재 다음 실행은 이전 학습 checkpoint 없이 전체 val500 평가 시간을 측정하는 v2**입니다.
[고정 설정](configs/val500_timing_initial_v2.json)을 사용합니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_val500_timing_initial.sh
```

- 기존 Tiny와 같은 **공개 AugReg ImageNet-21k→1k 사전학습 encoder + 학습 전 decoder**를
  seed1로 구성합니다. 이전 Cityscapes `resume.json`이나 학습 가중치는 필요하지 않습니다.
  새 학습·optimizer update는 0회이며 teacher와 guidance module을 생성하지 않습니다.
  teacher 사전학습 파일도 다운로드하지 않습니다. 본 증류 실험의 OpenMMLab teacher 선택은 그대로입니다.
- 공개 Tiny encoder NPZ는 23,226,422 bytes,
  SHA-256 `4b99893dc1a5a2a7d9ad119671c20559850f865e1fa17ed23401a3fefa7fedc9`로 검사합니다.
  고정 upstream 코드와 공식 가중치는 캐시에 없으면 내려받습니다.
- 데이터는 `CITYSCAPES_CROP512_DATA_DIR`(기본
  `/app/scratch/cityscapes_l16_crop512_v3/cityscapes`)의 manifest와 val 파일을 재사용합니다.
  manifest가 없으면 `CITYSCAPES_ZIP_DIR`(기본 `/app/data/chaoyang`)의
  `leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`을 기존 업로드 검사와 같은 byte/SHA로
  확인한 뒤 train/val을 압축 해제하고 감사합니다. 기존 파일 내용이 다르면 덮어쓰지 않고 실패합니다.
  원본 데이터 다운로드나 test 압축 해제는 하지 않습니다. val 실제 이미지/정답의 byte/SHA는 평가 전에 검사합니다.
- 기존 Tiny·decoder1·FP32·원본 해상도·window/stride512·window batch1·CPU thread4를 유지합니다.
  공식 fine val 500장을 정렬된 순서대로 딱 한 번 평가하고 test는 사용하지 않습니다.
  가짜 데이터나 해상도 축소로 시간을 추정하지 않습니다.
- `validation_seconds`는 첫 이미지 초기 비용·이미지 읽기·변환·전송·추론·지표 집계를 포함합니다.
  SHA 검사에서 파일을 먼저 읽으므로 OS 파일 캐시가 어느 정도 채워진 조건의 측정입니다.
  반복 학습 사이에 수행하는 같은 평가 코드의 시간 예산에 사용하며 cold-cache 최악 시간은 아닙니다.
- `total_job_seconds`에는 실행 스크립트 시작부터 설치·asset 준비·데이터 감사·모델 로딩·평가까지
  포함합니다. 큐 대기·외부 컨테이너 생성·git clone은 포함하지 않습니다.
- 25장마다 진행 상황을 출력하고 마지막 **`[CITYSCAPES_TI16_VAL500_TIMING_FINAL]`**에
  평가/전체 시간, 처리속도, 메모리, accuracy·mIoU·19 class IoU·유효 픽셀 수,
  `selected_step=0`, `selected_epoch=0`, `optimizer_updates=0`,
  `trained_checkpoint_loaded=false`, 가중치 무변화 및 실행 상태를 출력합니다.
  학습 loss/CE/guidance·β·λ·checkpoint는 `null`입니다.
  **accuracy·mIoU·IoU는 학습 전 decoder의 진단값입니다. 학습한 Vanilla 성능이나 후보 순위에 쓰지 않습니다.**
  동일 구조·입력·평가 경로의 소요 시간을 측정하여 후속 2k 작업의 시간 예산에 사용합니다.
  실제 학습 도중의 GPU 부하·파일 캐시 상태에 따라 시간이 달라질 수 있으므로 여유를 둡니다.

출력은 `/app/output/cityscapes_ti16_val500_timing_initial_v2/run_<UTC>_<PID>/`의 `summary.json`,
`terminal_summary.log`, `per_image_timings.json`, `validation_ids.json`, `warnings.json`,
`asset_provenance.json`, `data_setup.json`입니다. `CITYSCAPES_TI16_VAL_OUTPUT`으로 출력 위치를 지정할 수 있습니다.
설치 실패도 마지막 JSON으로 보고하며, 강제 종료로 출력 기회가 없었던 경우까지 보장하지는 않습니다.
실제 H200 val500 시간은 아직 측정되지 않았으며 이 실행 결과로 확인합니다.
새 경로 검사 6개와 기존 관련 검사 70개, 총 **로컬 검사 76개 통과**했습니다.
checkpoint·teacher·optimizer 없이 평가하는 분기, ZIP 손상 시 추출 전 중단,
기존 데이터 재사용, 평가 조건 일치, 초기 가중치 무변화 및 설치 실패 종료 JSON을 검사했습니다.
이 검사는 CPU의 작은 모델과 모의 GPU 경로를 포함하며 실제 H200 평가 시간을 대신하지 않습니다.

이전 checkpoint 평가 [v1 설정](configs/val500_timing_v1.json)과 `run_val500_timing.sh`는 기록용으로
보존합니다. 사용자 제공 #837 로그는 이전 컨테이너의 `lg_b1/resume.json`이 없어 preflight에서
실패했고, 평가 0장·학습 update 0회였습니다. 500step 학습 결과 자체가 실패한 것은 아닙니다.
v1은 여전히 검증된 checkpoint를 요구하며, v2로 자동 전환하지 않습니다.

**아래는 완료한 500-step 후보 검사 v6(24개)의 실행 기록**입니다.
2026-09-26 사용자 결정에 따라 **iBKD는 λ=0.25와 λ=0.5를 항상 함께 구성**합니다.
LG 8개 + iBKD λ0.25 8개 + iBKD λ0.5 8개이며,
[고정 config](configs/beta_grid500_shared_lg_8betas_v6.json)의 24개를 순서대로 실행합니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_beta_grid500.sh
```

LG는 **LG·ALG 공통 가이던스 구간의 사전 검사**입니다. 같은 β·초기값·입력에서 두 방법의
학습 연산은 같고, ALG는 가장 빨라도 744step(2epoch 완료)에 종료를 결정해 745step부터
가이던스를 끕니다. 500step에서는 ALG를 중복 실행하지 않으며 별도의 ALG 결과를 만들지 않습니다.
2k 단계에서는 LG와 ALG를 각각 구성해야 합니다. LG 체크포인트의 controller 종류를 이름만
바꿔 ALG로 재개하지 않습니다.

| 후보 | 초기 목표 비율 | LG | iBKD λ0.25 | iBKD λ0.5 |
|---|---:|---:|---:|---:|
| 1 | 1.5% | 0.00932921 | 0.0246183 | 0.0366835 |
| 2 | 3% | 0.0186584 | 0.0492367 | 0.0733669 |
| 3 | 4.5% | 0.0279876 | 0.073855 | 0.11005 |
| 4 | 6% | 0.0373168 | 0.0984734 | 0.146734 |
| 5 | 9% | 0.0559752 | 0.14771 | 0.220101 |
| 6 | 12% | 0.0746336 | 0.196947 | 0.293468 |
| 7 | 18% | 0.11195 | 0.29542 | 0.440202 |
| 8 | 24% | 0.149267 | 0.393893 | 0.586936 |

표는 표시용 6자리 유효숫자이며 실제 β는 JSON의 전체 정밀도를 사용합니다.
#834 초기 CE/guidance로 계산한 β₀에 `[0.5,1,1.5,2,3,4,6,8]`을 곱합니다.
기존 3/6/12/24% 후보에 1.5/4.5/9/18%를 추가했습니다. 초기 손실 크기에 대한 탐색 기준이며
학습 내내 유지되는 비율이나 논문의 표준값은 아닙니다. β를 재계산하지 않습니다.
추가 25-step 학습·calibration 없이 바로 500 update하며 Vanilla/FSKD*/C2VKD*는 실행하지 않습니다.
모델·데이터·학습 조건은 v4와 동일하고, 매 step 입력 해시는 기록하되 전체 gradient 해시는 계산하지 않습니다.

- `train2975`, 실제 batch8, 마지막 batch7을 포함해 **372 step=1epoch**입니다.
  500step은 1epoch 완료 + 2epoch의 128 batch, 총 3,999 sample 관측입니다.
  같은 데이터를 다른 epoch에서 반복 관측한 수이며 고유 이미지 수는 아닙니다.
- 완료 epoch의 실제 sample 수로 가중 평균한 guidance만 controller에 전달합니다.
  ALG warm-up0, iBKD20epoch 조건은 유지하며 부분 epoch를 완료 처리하지 않습니다.
- SGD LR0.01 및 **80,000-step schedule**을 유지하고 500에서 멈춥니다.
  다음 2k/10k/80k를 자동 실행하지 않습니다.
- NaN/Inf loss·gradient·parameter/optimizer state는 해당 후보를 중단합니다.
  OOM·입력 오류 등은 별도 runtime failure입니다. 낮은 진단 mIoU나 유한한 loss 급등만으로
  자동 중단하지 않으며, 실패 후보를 영구 제외하거나 우수 후보를 자동 선정하지 않습니다.
- 후보마다 별도 프로세스에서 새로 학습합니다. 실패해도 나머지 후보를 계속 실행합니다.
- 끝에서 고정 val2장의 accuracy·mIoU·19 class IoU를 계산합니다. **전체 val500 평가가 아니며
  성능 순위 선정용이 아닙니다.** 2k 단계의 전체 val 비교는 별도 구성해야 합니다.

출력: `/app/output/cityscapes_ti16_beta_grid500_shared_lg_8betas_v6/run_<UTC>_<PID>/`.
마지막 **`[CITYSCAPES_TI16_GRID500_FINAL]`** JSON은 24개 전체 결과를 담고,
접두사·줄바꿈을 포함해 **50,000 ASCII bytes 이하**로 제한합니다. 따라서 문자 수도 같습니다.
서버가 마지막 65,000자를 제공하면 후속 출력에 15,000자 여유가 있습니다.

- 각 후보의 β·λ·초기 목표 비율·완료/시도 step·선택 step/epoch·최종 loss/CE/guidance·
  alignment/fusion·가중 guidance/CE 비율·gradient norm·LR을 출력합니다.
- 진단 accuracy·mIoU·19 class IoU·평가 클래스/픽셀 수·가이던스 종료·teacher 고정·
  checkpoint 저장 step/검사·경고 횟수·실패 이유 요약·시간/메모리도 포함합니다.
- 초기/마지막25 중앙값·최대·최소는 `trajectory_order`에 표시된 순서입니다.
  클래스별 IoU 배열은 `class_iou_order`의 19개 클래스 순서입니다.
- 출력 수치는 6자리 유효숫자이며 β는 전체 정밀도를 유지합니다. 긴 경고 원문·상태 이력·
  파일별 SHA는 로그에서 반복하지 않고 결과 파일에 보존합니다. 드문 크기 초과 시에는
  `run_columns`와 행 배열로 키를 공유해 모든 후보와 수치 필드를 유지합니다.
- 평가하지 못한 수치는 null, 아직 시작하지 못한 후보는 `not_run`입니다.
  설치 실패 시에도 24개 계획값과 미실행 상태를 출력하며 결과를 만들어내지 않습니다.
- `artifacts/grid_summary.json`에는 원래 정밀도의 종합 결과와 경고를,
  후보별 `summary.json`/`steps.jsonl`/`warnings.json`에는 상세 이력을 저장합니다.
  `artifacts/terminal_summary.log`는 최종 출력 복사본입니다. Shell 실패 처리 시 출력 루트에도 남깁니다.

24개 모두 finite 500step 완료 및 공통 초기값/입력 비교를 통과하면 `passed`,
실패 후보나 비교 불일치가 있으면 `needs_review`와 전체 결과를 남깁니다.
강제 종료로 프로세스가 로그를 쓸 기회가 없었던 경우까지 종료 JSON을 보장하는 뜻은 아닙니다.

각 후보의 `resume.json`과 `checkpoints/`는 최신·이전 두 세대를 보관합니다.
0/250/epoch 경계/500step에 모델·guidance·optimizer·scheduler·controller·RNG·입력 진행 위치·
부분 epoch 누적값을 저장하고 bytes/SHA를 기록합니다. 완료 시 저장 상태를 엄격 비교합니다.
중단된 동일 config/코드 실험은 `tiny_grid --run-id <id> --resume <resume.json>`으로
새 출력 폴더에서 500까지 이어갈 수 있습니다. Dataset/asset 검증을 동일하게 수행한 환경에서 사용합니다.
후속 2k 전환은 **새 단계의 프로토콜과 재개 호환성을 먼저 고정**해야 하며,
현재 CLI의 config identity 검사를 임의로 우회하지 않습니다.

v6 관련 로컬 검사 **63개 통과**. 24개 전체 지표·긴 경고/오류를 넣은 성공/부분 실패/전체 실패
종료 로그는 각각 **36,876 / 38,945 / 41,010 bytes**였습니다.
마지막 65,000자만 남겨도 24개 결과를 모두 읽을 수 있는지 확인했습니다.
에폭 경계·부분 epoch 재개·후보 실패 후 나머지 실행·설치 실패 출력도 검사했습니다.
이는 실제 H200 500step 결과를 미리 보장하는 의미가 아닙니다.

이전 16개 구성은 [v5 config](configs/beta_grid500_both_lambdas_v5.json)에 보존합니다.
기존 이슈의 고정 commit `3e3d4b93188a9b474d4ad2fc76c8cf6140423c18`은 v5 그대로 실행되므로
24개 실행은 새 commit의 이슈 명령을 사용해야 합니다.

**이전 재현성 진단 v4는 H200에서 통과했습니다.** LG-A/LG-B/ALG의 25step 입력·RNG·logits·
gradient·update 후 state 해시가 모두 동일했고, CE scalar의 약 7.15e-7 이하 차이는 허용 오차 내였습니다.
아래는 해당 v4 실행 기록입니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_repeat25_lg_alg.sh
```

[repeat25_lg_alg_v4.json](configs/repeat25_lg_alg_v4.json)에 고정한 세 경로는
`lg_a`, `lg_b`, `alg`이며 각각 별도 Python 프로세스에서 실행합니다.
β는 #834 LG의 초기 제안값 `0.018658411532808426`으로 **모두 완전히 동일하게** 사용합니다.
초기 25 batch 손실 측정은 유지하되 β를 다시 계산하거나 선택하지 않고, 상태를 복원한 뒤
각각 25 update합니다. Tiny/teacher·crop512·batch8·seed1·SGD·증강·FP32 및
80k schedule은 기존과 같습니다. iBKD/FSKD/C2VKD는 이번 진단에서 실행하지 않습니다.

각 step에 입력, Python/NumPy/Torch CPU·CUDA RNG, logits, 전체 student/adapter gradient,
update 후 state의 SHA-256을 기록합니다. 이러한 진단 때문에 이번 step 시간은 일반 학습
속도 추정에 쓰지 않습니다. `warn_only=True`를 유지해 경고가 나도 관찰을 계속하고,
경고 원문·횟수·소스 위치를 성공/실패 모두 마지막 JSON에 남깁니다.

최종 `repeat_comparisons.pairs`에는 LG-A↔LG-B, LG-A↔ALG, LG-B↔ALG **모두** 기록됩니다.
최초 scalar 차이 step·양쪽 값·허용 오차·최대 차이 및 항목별 최초 hash 차이 step을 표시합니다.
입력/초기값/β/RNG가 동일해야 하며, scalar 비교는 기존 `rtol=2e-5, atol=2e-6`를 유지합니다.
출력/gradient/state의 hash 일치는 별도 엄격 진단입니다. Hash가 다르지만 scalar 오차가
허용 범위 내인 경우에도 bitwise 재현이라고 해석하지 않습니다.
세 실행과 비교가 모두 통과해야 전체 `status=passed`입니다. 차이가 나면 전체 failed와
세 쌍의 진단을 남기고, 이후 500/2k/10k 학습을 자동 실행하지 않습니다.

각 실행의 loss·선택 step/epoch·진단 val2 accuracy/mIoU/IoU·재개 검사도 기존처럼 출력합니다.
25step은 1epoch를 채우지 않아 자연스러운 controller 종료는 검사하지 않습니다.
출력 위치: `/app/output/cityscapes_ti16_crop512_smoke25_repeat_v4/run_<UTC>_<PID>/`.
경로별 `summary.json`/`training_progress.json`에 모든 step trace,
`diagnostic_initial.json`에 초기값, `warnings.json`에 경고를 남깁니다.
`artifacts/smoke_summary.json`과 마지막 `[CITYSCAPES_TI16_SMOKE_FINAL]`에 전체 판단이 있습니다.

**아래는 이전 C2VKD* 추가 v3(7경로)의 기록**입니다.
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

v4 로컬 검사: 관련 단위 검사 52개 통과. 실제 공식 Tiny NPZ와 32×32 합성 입력·합성
teacher feature로 LG-A/LG-B/ALG를 각각 25 update해 loss, RNG, logits, gradient,
update 후 state의 모든 비교가 CPU에서 일치함을 확인했습니다. 실패 시 경고 원문과
종료 JSON 보존, 설치 실패 시 즉시 중단도 검사했습니다. H200에서의 반복 재현성은
이 v4 이슈의 결과로 확인해야 합니다.


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
