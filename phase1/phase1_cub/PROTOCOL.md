# Phase 1 CUB-200-2011 프로토콜

상태: **본실험 v1 LOCK — 실행 전**

이 문서는 CUB-200-2011 전용 실험 계약입니다. 아래 항목을
`configs/cub200_b128_full_v1.json`에 고정했으며, 결과를 확인한 뒤에는 v1을
수정하지 않습니다. 변경이 필요하면 새 버전을 만들고 비교하는 여섯 설정을 모두
다시 실행합니다.

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

## 잠긴 본실험 v1 계약

Full config SHA-256:
`86e23457579935f68841bf068b613379e2f96d49e1196875064049a506058374`

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

## 10시간 제한용 두 실행 shard

두 shard는 동일한 full config를 사용하고 각각 teacher를 seed 1로 재현합니다.
두 teacher state SHA-256, seed별 초기 student state SHA-256과 split hash가
일치해야 결과를 합칠 수 있습니다.

- Shard A: Vanilla, LG, iBKD λ=0.5 — 학생 9개, probe 후보 135개,
  선택·test probe 45개, smoke 외삽 약 7시간 47분
- Shard B: KD, ALG-w20, iBKD λ=0.25 — 학생 9개, probe 후보 135개,
  선택·test probe 45개, smoke 외삽 약 7시간 49분

각 shard는 이미지와 segmentation archive를 `/app/scratch`에 독립 다운로드하고,
분류 best checkpoint 9개, 선택된 probe checkpoint 45개, teacher checkpoint 1개와
CSV/JSON 결과 및 `run.log`를 `/app/output`에 남깁니다. 데이터셋과 feature cache는
결과 archive에 포함하지 않습니다.

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
