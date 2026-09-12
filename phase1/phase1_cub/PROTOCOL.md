# Phase 1 CUB-200-2011 프로토콜

상태: **본실험 v3 LOCK — batch128 guided 4방법×3seed 및 직접 공간진단 3seed 실행·감사 완료**

현재 학생 이미지 loader 자체를 점검하는 사후 탐색 절차는
[LOADER_EXPERIMENT_PROTOCOL.md](LOADER_EXPERIMENT_PROTOCOL.md)에 별도 버전으로
고정했습니다. 이 절차는 완료된 v3 결과를 수정하지 않으며, loader 선택 전 official
test를 다시 열지 않습니다.

이 문서는 CUB-200-2011 전용 실험 계약입니다. 현재 항목은
`configs/cub200_r50_224_b128_full_v3.json`에 고정했습니다. 결과와 관계없이
여섯 설정 × 세 encoder seed를 모두 수행하며, smoke나 official test 수치를 보고
설정을 바꾸지 않습니다. 변경이 필요하면 현재 실행을 별도 pilot으로 분리하고
버전을 올린 뒤 비교하는 여섯 설정을 모두 다시 실행합니다.

분류→frozen segmentation probe 이후의 직접 공간정보 진단은 결과를 보기 전에
[DIRECT_SPATIAL_PROTOCOL.md](DIRECT_SPATIAL_PROTOCOL.md)에 v1으로 고정했습니다.
Part localization PCK@0.1을 주 직접 지표로, spatial linear CKA와 attention–GT를
보조 지표로 사용합니다. Issue 737의 seed 1과 issue 739의 seed 2·3이 같은 v2
정의로 완료됐으며, 독립 encoder seed 3개 집계까지 감사를 마쳤습니다.

## 잠긴 본실험 v3 계약

Full config SHA-256:
`e3faff49101a8cffc5d0836f2cf299177547cea5243715ce51cc288b743626dc`

- split은 train 5,394 / validation 600 / official test 5,794이며 v2와 같습니다.
- teacher는 TorchVision ResNet-50, scratch, 224×224, seed 1, batch 128,
  200 epoch입니다. SGD(lr 0.05, momentum 0.9, weight decay 1e-4,
  Nesterov 없음), 5-epoch linear warm-up 뒤 총 200-epoch cosine을 사용합니다.
- teacher train view는 RandomResizedCrop 224(bicubic)+horizontal flip,
  evaluation view는 shorter-side 256 resize+center crop 224입니다.
- teacher feature는 `layer2/layer3/layer4`, channel `[512,1024,2048]`, grid
  `[28,14,7]`입니다. 모든 guided student가 validation으로 선택한 teacher
  checkpoint 하나를 공유합니다.
- student, LG, ALG-w20, iBKD, frozen probe의 값은 v2와 같습니다. Teacher
  channel projection만 ResNet-50 feature contract에 맞춥니다.
- 최종 비교는 Vanilla, KD, LG, ALG-w20, iBKD λ=0.25, iBKD λ=0.5의
  6설정 × encoder seed `[1,2,3]` 전부이며 결과에 따른 중단·방법 제외·lambda
  선택을 하지 않습니다.
- 분류 checkpoint와 probe checkpoint 선택은 validation만 사용합니다.
  Official test는 선택된 각 checkpoint를 한 번 평가하며 method, lambda, LR,
  epoch 선택에는 사용하지 않습니다.

## v3 Teacher 전용 본학습 실행

- 실행기: `scripts/run_r50_224_teacher_full.sh`
- 구현: `ibkd_seg.phase1.run_cub_r50_teacher_full`
- 기존 ResNet-56/32·300 epoch용 `ibkd_seg.phase1.train_full`은 사용하지 않습니다.
- Full config의 byte-level SHA-256과 scratch/224/batch 128/200 epoch/optimizer/
  scheduler/split/test 규칙을 모두 실행 전에 검사하며, 하나라도 달라지면 학습을
  시작하지 않습니다.
- validation macro top-1 최대 checkpoint 하나만 선택하고 동률이면 이른 epoch를
  유지합니다. 선택한 파일을 새 ResNet-50에 strict reload하고 model-state hash를
  검증한 뒤 official test를 한 번만 평가합니다.
- 산출물의 checkpoint SHA-256과 model-state SHA-256을 후속 LG, ALG-w20,
  iBKD 학생 실행기에 함께 고정해 모든 guided 방법이 동일 Teacher를 사용하게 합니다.

H200 issue 722 결과는 위 계약대로 완료되고 독립 감사를 통과했습니다. 선택 epoch는
165, validation macro top-1은 `42.667%`, official-test macro top-1은
`41.178%`입니다. 공유 Teacher checkpoint의 파일 SHA-256은
`ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3`,
model-state SHA-256은
`96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7`입니다.

## v3 guided end-to-end smoke 계약

Smoke config SHA-256:
`f1239e1533f40fce10bd2f5675c19284f26d91c804709b480de1dac745a2d1a4`

- teacher 1개와 LG, ALG-w20, iBKD λ=0.25/0.5 네 student를 seed 1에서
  full train 5,394장으로 각각 2 epoch 실행합니다.
- 각 2-epoch checkpoint의 classification validation과 official test를 각각
  평가합니다.
- 네 encoder를 strict load·완전 동결하고 LR 3개 × 2-epoch probe를 수행한 뒤,
  validation으로 선택한 probe를 official test에 정확히 한 번 평가합니다.
- official test 실행은 최종 6방법×3seed와 모든 설정을 결과와 무관하게 수행한다는
  사전 확정 때문에 허용합니다. 모든 smoke metric은 여전히 비과학적이며 논문
  결과나 설정 선택에 사용할 수 없습니다.
- batch 128 FP32 peak memory와 단계별 시간, 4방법×3seed guided block의 선형
  외삽을 기록합니다. Smoke가 OOM이면 batch나 precision을 바꾸지 않고 필요한
  MIG 용량만 늘려 동일 설정을 다시 실행합니다.

Smoke는 Full H200에서 `17/17`로 통과했습니다. Teacher 측정값은 epoch당 약
7.56초, peak allocated 11.377 GB, peak reserved 15.731 GB였습니다. 실제 Teacher
본학습은 25분 42초에 완료됐고 동일한 peak memory를 기록했습니다.

### Batch 128·64 비과학적 profile smoke

후속 실행시간과 메모리를 같은 Teacher 조건에서 비교하기 위한 v4 smoke config는
`configs/cub200_r50_224_b64_b128_guided_smoke_v4.json`이며 SHA-256은
`dd8e61c94f085096fed18615af217dd9b9e35bda0d79bdb19154d168c92ef211`입니다.
Issue 722 Teacher를 다시 학습하지 않고 release에서 받아 checkpoint SHA-256
`ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3`와
model-state SHA-256
`96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7`을 확인합니다.
그 하나를 네 guided 방법과 두 student batch가 모두 공유합니다. Batch 64는
비과학적 sensitivity profile이며 이 smoke는 잠긴 batch-128 v3 계약을 변경하지
않습니다.

### Batch 128·64 guided seed-1 full-epoch profile LOCK

Full profile config는
`configs/cub200_r50_224_b128_b64_guided_seed1_full_v4.json`이며 SHA-256은
`bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633`입니다.

- 범위는 LG, ALG-w20, iBKD λ=0.25/0.5 × encoder seed 1 × student batch
  `[128,64]`입니다. Teacher는 issue 722의 동일 checkpoint 하나입니다.
- 학생은 300 epoch, probe는 encoder마다 5 seed × LR 3개 × 100 epoch입니다.
- Batch 128은 v3 주 매트릭스의 일부 셀이고 batch 64는 sensitivity입니다. 전체
  6방법×3seed 완료로 해석하지 않습니다.
- 각 batch의 probe 20개를 validation으로 모두 선택한 뒤에만 그 batch의 official
  test mask를 열며, test는 선택이나 설정 변경에 사용하지 않습니다.
- 분류 best checkpoint 8개와 선택 probe checkpoint 40개를 보존합니다.

H200 issue 727은 이 계약을 변경하지 않고 완료됐습니다. 분류 `8/8`, probe LR
후보 `120/120`, validation 선택·official test `40/40`이며 새 checkpoint 48개가
모두 파일 hash, strict load와 유한값 감사를 통과했습니다. Seed-1 수치는
[v4 부분 결과 보고서](reports/frozen_probe/resnet50_224_b128_b64_guided_seed1_v4/RESULTS.md)에
고정했습니다. 이 결과를 보고 v5 범위나 방법·lambda를 바꾸지 않습니다.

### Guided encoder-seed 후속 범위 LOCK

Seed-1 profile 로그를 확인한 뒤 다음 후속 범위를 고정했습니다. Full config는
`configs/cub200_r50_224_b128_s23_b64_s2_guided_full_v5.json`, SHA-256은
`f3531c648f65e6f51e48bbeda7ad38b1fc5931d88e04b01c97c6ff71aad437b9`입니다.

- Batch 128은 guided 네 방법의 encoder seed 2·3을 추가합니다. 기존 seed 1과
  합쳐 `[1,2,3]`이 되며 잠긴 v3 주 매트릭스의 일부로 사용할 수 있습니다.
- Batch 64는 encoder seed 2만 추가합니다. Seed-1 결과 확인 뒤 결정했으므로
  `[1,2]` 모두 탐색적 batch sensitivity이고 확증적 주장에 사용하지 않습니다.
- Teacher, split, 학생 초기화 방식, 300 epoch 학습값, ALG/iBKD controller,
  lambda 두 개, validation checkpoint 선택 및 test-once 규칙을 변경하지 않습니다.
- 각 encoder에는 probe seed 5개 × LR 3개 × 100 epoch를 동일하게 수행합니다.
  Probe seed를 독립 encoder 실행처럼 펼치지 않고 encoder 내부에서 먼저 평균냅니다.
- 이 범위는 guided 네 방법만 포함하므로 최종 Vanilla/KD 포함 6방법×3seed
  매트릭스 완료로 표시하지 않습니다.
- Part localization, spatial CKA 및 attention–GT 진단은 checkpoint archive 회수
  뒤 별도 프로토콜로 고정하며 이번 분류→probe 실행에는 포함하지 않습니다.

선행 비과학적 smoke config는
`configs/cub200_r50_224_b128_s23_b64_s2_guided_smoke_v5.json`, SHA-256은
`6160cdcb19f2225e574bf9f397b1c99be27c95c006ba7dfc88d9e41344042162`입니다. 정확히
`(128,2)`, `(128,3)`, `(64,2)`만 허용하고 각 profile에서 guided 네 방법 ×
2-epoch 분류와 probe seed 1 × LR 3개 × 2 epoch를 검사합니다. 같은 encoder seed
내 네 방법의 초기 state와 seed 2의 두 batch 간 초기 state가 같고, seed 2와 3의
초기 state는 다른지 hash로 검증합니다. 모든 smoke metric은 비과학적입니다.

Smoke는 profile `3/3`, 분류 `12/12`, probe 후보 `36/36`, 선택 probe `12/12`,
GPU 작업 `48/48`로 통과했습니다. 고정한 H200 분할 중 batch 128 seed 2·3은
`scripts/run_r50_224_guided_probe_full_b128_seeds2_3.sh`로 함께 실행합니다. 이
분할의 새 산출물은 분류 checkpoint 8개와 선택 probe checkpoint 40개이며, 마지막
완료 gate는 `classification=8/8`, `probe_candidates=120/120`, `selections=40/40`,
`test_once=40/40`, `new_checkpoints=48`입니다. Smoke 선형 외삽은 약 9시간 4분이고
데이터 다운로드·최초 target cache overhead는 별도이므로 10시간 제한에 가깝습니다.

H200 issue 730은 7시간 25분 54초에 위 완료 gate를 모두 충족했습니다. Issue 727의
seed 1과 합쳐 batch-128 guided 네 방법의 encoder seed `[1,2,3]`가 완성됐고 3-seed
집계는 [v5 결과 보고서](reports/frozen_probe/resnet50_224_b128_guided_3seed_v5/RESULTS.md)에
고정했습니다. 이 결과를 보고 Vanilla/KD나 직접진단 설정을 변경하지 않습니다.

## 고정한 항목

- 공식 이미지·분류 annotation·segmentation mask의 출처, byte size, SHA-256
- 이미지 ID와 class label, 공식 split, mask의 1:1 대응 및 누락 처리
- train/validation/test 분할과 test-once 정책
- pixel class 정의, resize/interpolation, 경계 또는 무효 픽셀 처리
- CNN teacher와 ViT student 구조 및 초기화
- Vanilla, KD, LG, ALG, iBKD의 공통 설정과 방법별 고정값
- batch size, epoch, optimizer, learning-rate schedule과 seed
- 분류 checkpoint의 validation-only 선택 기준
- frozen feature 위치·shape·normalization과 encoder freeze 검사
- 공통 probe 구조, LR 후보, epoch, seed와 validation 선택 기준
- IoU, mIoU, Dice 등 최종 metric과 비영상 baseline
- smoke와 본실험의 분리, 산출물 manifest와 결과 보고 형식

## Pet과의 관계

Pet은 완료된 독립 실험이므로 CUB 결과에 맞춰 Pet의 LOCK config나 결과를
수정하지 않습니다. 가능한 조건은 Pet과 맞추되, CUB 데이터 특성 때문에 달라지는
항목은 근거와 함께 이 문서에 명시합니다.

## 대체된 본실험 v2 계약과 회수 결과

아래 ResNet-56/32 v2 guided shard는 v3 교체 전에 제출되어 H200 issue 716에서
완료됐지만, 결과를 회수하기 전에 v3를 최종 프로토콜로 확정했습니다. 따라서 v2
baseline shard는 실행하지 않고, 회수 결과를 v3와 비교하거나 합치지 않습니다.

Full config SHA-256:
`0cf751c28168872a4108274644f80dadc7466d5c1210995e7da3abfc0737e575`

- CaltechDATA 이미지 archive: 1,150,585,339 bytes, MD5
  `97eceeb196236b17998738112f37df78`, SHA-256
  `0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081`
- CaltechDATA segmentation archive: 39,272,883 bytes, MD5
  `4d47ba1228eae64f2fa547c47bc65255`, SHA-256
  `dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f`
- 공식 train 5,994장 중 클래스별 3장, 총 600장을 seed 2027 validation으로
  고정하고 train 5,394장을 학습에 사용합니다. Validation ID SHA-256은
  `263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854`입니다.
- 공식 test 5,794장은 validation 선택에 사용하지 않습니다.
- scratch ResNet-56 teacher(seed 1) 1개와 scratch DeiT-Tiny student seed
  `[1, 2, 3]`, batch 128, 300 epoch를 사용합니다.
- 비교 설정은 Vanilla, KD, LG, ALG-w20, iBKD λ=0.25, iBKD λ=0.5입니다.
  Canonical ALG warm-up 0은 포함하지 않으며 두 iBKD λ를 모두 별도로 보고합니다.
- 분류 checkpoint는 validation macro top-1 최대값으로 선택하고 동률이면 이른
  epoch를 사용합니다. 선택한 checkpoint만 frozen probe로 전달합니다.
- CUB binary mask는 grayscale `0=background`, `>0=foreground`로 사용하고
  무효·경계 pixel을 별도로 만들지 않습니다.
- 완전히 동결한 DeiT block 11의 `192×14×14` feature에 공통
  `Conv2d(192,2,1)` probe만 학습합니다.
- Probe는 batch 64, 100 epoch, seed `[1,2,3,4,5]`와 LR
  `[0.01,0.03,0.1]`을 사용합니다. Validation grid mIoU 최대값으로 LR와 epoch를
  선택하고 동률이면 낮은 LR, 이른 epoch 순서로 결정합니다.
- 공식 test mask는 한 shard의 45개 validation 선택이 모두 끝난 뒤 처음 decode하며,
  선택된 각 probe를 한 번씩만 평가합니다.
- 독립 반복 단위는 encoder seed입니다. Probe seed는 encoder 반복 수로 세지 않고,
  encoder별 5개 probe 평균을 낸 뒤 3개 encoder seed의 평균과 표본표준편차를
  보고합니다.
- 정성 예시는 결과 확인 전 test image ID
  `[787, 2285, 3735, 5205, 6691, 8139, 9597, 11064]`로 고정했습니다.

## 대체된 v2의 10시간 제한 및 단일 teacher용 두 단계

다음은 v3 확정 전 v2 실행 계획과 실제 완료 상태를 보존한 기록입니다. 두 shard는
동일한 v2 full config를 사용하되 teacher를 두 번 학습하지 않도록 설계했습니다.

- Guided producer: teacher seed 1을 한 번 학습한 뒤 ALG-w20, iBKD λ=0.25,
  iBKD λ=0.5의 학생 9개와 probe 후보 135개, 선택·test probe 45개를 실행합니다.
  Smoke 핵심 연산 외삽은 약 7시간 59분이고 데이터·feature/test overhead를 더한
  예상은 약 8시간 10~30분입니다.
- Baseline consumer: guided 결과에서 고정한 정확히 같은 teacher checkpoint를
  사용해 Vanilla, KD, LG의 학생 9개와 probe를 실행합니다. Teacher를 재학습하지
  않으며 핵심 연산 외삽은 약 6시간 54분입니다.

Guided shard는 9개 분류 encoder, 135개 probe 후보, 45개 validation 선택과 45개
official-test 평가를 완료했고 55개 checkpoint의 독립 감사도 통과했습니다.
구버전 조건의 test input-224 mIoU는 ALG-w20 `77.882%`, iBKD λ=0.25
`77.546%`, iBKD λ=0.5 `77.538%`였습니다. 이는 v2 참고 결과이며 v3 핵심 주장
판정에는 사용하지 않습니다.

Guided 실행은 이미지와 segmentation archive를 `/app/scratch`에 다운로드했고,
분류 best checkpoint 9개, 선택된 probe checkpoint 45개, teacher checkpoint 1개와
CSV/JSON 결과 및 `run.log`를 `/app/output`에 남겼습니다. 데이터셋과 feature
cache는 결과 archive에 포함하지 않았으며, v2 baseline 실행기는 추가하지 않습니다.

## 본실험 전 통합 smoke 계약

본실험 결과를 보기 전 실행 경로를 검증하기 위한 batch-128 통합 smoke만
`configs/cub200_b128_combined_smoke_v2.json`에 별도로 고정했습니다. 이는 이
문서의 본실험 LOCK을 대신하지 않습니다.

Smoke config SHA-256:
`502e6d5e285452d1aeae3eb1ec3e912da34337d2bfc6e3bdb1868ce444941699`

실행되지 않은 v1은 CUB의 ALG warm-up 0 조건을 제거하면서 v2로 대체했으며,
과거 내용은 Git 이력에만 보존합니다.

- 공식 train 5,994장만 클래스별 고정 분할: train 5,394 / validation 600
  (클래스당 3장, seed 2027)
- 공식 test 5,794장: loader 생성·이미지/mask decode·평가 모두 금지
- scratch CIFAR-style ResNet-56 teacher 1개, 입력 32, batch 128, seed 1,
  2 epoch
- scratch DeiT-Tiny student 6개: Vanilla, KD, LG, ALG-w20, iBKD λ=0.25/0.5,
  입력 224, batch 128, seed 1, 각각 2 epoch
- CUB에서는 ALG controller 종료 판정 warm-up을 20 epoch로 사전 고정하고
  canonical warm-up 0 조건은 제외; 결과 표기명은 `ALG-w20`
- iBKD controller 종료 판정 warm-up 20
- 각 student 최신 smoke checkpoint를 strict load하고 encoder를 완전히 동결
- 공식 binary mask의 grayscale 값 `> 0`을 foreground로 임시 고정하고 nearest
  resize
- 1×1 probe, seed 1, LR `[0.01, 0.03, 0.1]`, 후보별 2 epoch, validation만 평가
- 모든 smoke 정확도·IoU는 비과학적 진단값이며 본실험 선택과 논문 주장에 사용 금지

Smoke는 `25/25`로 통과했습니다. Smoke 수치는 선택에 사용하지 않았고, 위 full
config를 별도로 LOCK했습니다.
