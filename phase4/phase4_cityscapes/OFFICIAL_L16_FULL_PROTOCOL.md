# Cityscapes 공개 소스 L/16 본학습 v1

2026-09-16 사용자 요청에 따라 본학습 조건을 고정합니다.
설정 파일은 `configs/official_l16_full_v1.json`입니다.
H200 `bapedragon_771`의 공식 소스·공개 가중치 smoke가 4/4 통과한 뒤 준비한
**seed 1의 네 방법 비교**입니다. 여러 seed의 평균·표준편차 실험은 아닙니다.

## 비교 기준과 범위

기준은 [Segmenter 원본 저장소](https://github.com/rstrudel/segmenter/tree/20d1bfad354165ee45c3f65972a4d9c131f58d53)의
`config.yml` 및 `train.py` 기본값입니다. 원본 모델·증강·optimizer·scheduler·inference를
직접 호출하고 원본 파일은 수정하지 않습니다.

Cityscapes 배포 모델의 `variant.yml`은 2026-09-16 확인 시 HTTP403이었고,
코드 기본 구성은 334,845,966 parameters, README의 Cityscapes 표는 322M입니다.
따라서 **배포 논문 결과의 세부 설정까지 일치하는 완전 재현이라고 표기하지 않습니다.**
공개 코드 기본값을 공통 기반으로 쓰고 LG·ALG·iBKD를 적용하는 연구 실험입니다.
LG·ALG·iBKD에 대해 Cityscapes 공식 기관이 지정한 학습 프로토콜이 있다는 뜻도 아닙니다.

## 고정 학습 조건

| 항목 | 값 |
|---|---|
| 데이터 | fine train 2,975장 / val 500장, 공식 19 trainIds, ignore255 |
| Teacher | MMSeg DeepLabV3 ResNetV1c-101-D8, Cityscapes 80k 공개 checkpoint, 전체 고정 |
| Student | 원본 ViT-L/16 24블록·1024채널 + 원본 2블록 mask transformer decoder |
| Student 초기화 | 저자가 지정한 AugReg ImageNet NPZ로 encoder 초기화, decoder 새 초기화 |
| 방법·seed | Vanilla, LG, ALG, iBKD 각각 seed1; student 초기 state 동일 |
| 기간 | 216epoch × ceil(2975/8) = 80,352 updates |
| Batch | 실제 8장, gradient accumulation 없음, 마지막 batch7 유지 |
| 학습 입력 | crop768, resize ratio0.5–2.0, category ratio0.75, flip·photometric distortion·pad |
| 정규화 | student ViT mean/std127.5; teacher에는 같은 증강 이미지를 ImageNet 정규화 |
| 최적화 | 원본 timm0.4.12 SGD Nesterov, lr0.01, momentum0.9, wd0 |
| LR | 원본 PolynomialLR, power0.9, min1e-5, 분모80,352, warmup0 |
| 정밀도 | FP32, AMP/TF32 없음, gradient clipping 없음 |
| 메모리 | transformer activation recomputation, iBKD 모든 key를 유지한 query chunk256 |

Teacher를 새로 학습하는 별도 작업은 없습니다. Smoke의 학습된 student state를
이어 쓰지 않고, 본학습은 위 공개 초기 가중치에서 새로 시작합니다.
모델 다운로드·해시 검증이 실패하면 무작위 초기화로 대체하지 않습니다.

## 방법별 고정값

- 공통 guidance beta2.5, CE + beta × guidance.
- LG: teacher stage2/3/4와 student 블록 `[0,12,23]`의 기존 LocalityGuidance. 216epoch 유지.
- ALG: LG와 동일한 guidance, window50, threshold−0.02, warmup0.
  기존 controller의 smoothed derivative가 threshold 이상이면 다음 epoch부터 beta0.
- iBKD: 24블록 전체의 학습 가능한 가중합 + 기존 alignment/fusion, fusion ratio0.25.
  window50, threshold−0.02, warmup20. threshold를 엄격히 초과하면 다음 epoch부터 beta0.
- Controller에 입력하는 epoch guidance loss는 batch별 loss를 실제 표본 수로 가중 평균합니다.
  마지막 batch7도 반영합니다. 한 epoch당 한 번만 관측합니다.
- 종료 후 student CE 학습은 216epoch까지 계속하고 teacher/guidance forward는 생략합니다.
  재개 시 활성 상태·loss/derivative/beta 이력·종료 epoch를 복원합니다.

Smoke의 LG/ALG loss 상승만으로 beta·LR·clipping·종료 규칙을 바꾸지 않습니다.
비유한 loss/gradient/parameter 발생 시 실패와 직전 저장 상태를 남기고 중단합니다.
실패한 방법도 비교 결과에 남기며 다른 설정으로 자동 재실행하지 않습니다.

## 평가와 checkpoint 선택

원본 코드의 `epoch % 4 == 0`은 0부터 세는 epoch입니다.
따라서 사람이 읽는 epoch **1,5,9,…,213과 마지막216**, 총55회 평가합니다.
매번 **val500장 전체**, 원본1024×2048, 단일 scale, 원본 window768/stride512 inference를 씁니다.

1. 전체 유효 픽셀의 **pixel accuracy**가 가장 높은 student checkpoint를 선택합니다.
2. 정확도가 같으면 먼저 선택한, 더 이른 epoch를 유지합니다.
3. **그 checkpoint의 mIoU·클래스별 IoU·confusion matrix**를 함께 기록합니다.
   mIoU 최고 epoch를 별도로 골라 주 결과에 섞지 않습니다.
4. ignore255는 정확도·IoU 분모에서 제외합니다. Test는 학습·선택·평가에 사용하지 않습니다.
5. 중단된 validation의 부분 점수는 선택에 사용하지 않습니다. 재개 시 해당 val500 평가를 다시 합니다.

Pixel accuracy 선택은 사용자가 정한 본실험 기준입니다. 공식 Cityscapes 대표 지표인
mIoU도 함께 보고하지만, validation 선택 점수는 독립된 test 성능이 아닙니다.

## 재현성과 저장·재개

원본 데이터 변환의 분포를 유지하면서, wrapper가 epoch별 permutation과
`seed/epoch/sample ID`별 증강 RNG를 따로 고정합니다. 네 방법에 같은 샘플 순서·증강을
제공하고, worker prefetch나 재개 위치 때문에 dropout RNG가 바뀌지 않게 합니다.
이는 저자 실행 당시의 난수 스트림을 그대로 복제한다는 뜻은 아닙니다.
CPU 데이터 worker4를 사용하며 재개 때 미처리 batch부터 읽습니다.

- 원본 ZIP·이미지/labelIds·변환 trainIds와 원본 소스3개·공개 가중치2개를 검증합니다.
  구체적인 URL/byte size/SHA-256은 기존 `official_assets.py`와 upload 검사기에 고정되어 있습니다.
- model·guidance·optimizer(momentum)·scheduler·controller·Python/NumPy/Torch/CUDA RNG,
  epoch/batch 위치·부분 loss 합계·best checkpoint 참조를 저장합니다.
- 최초, 100step마다, epoch 전환/평가 전후, 정상 일시정지에 저장합니다.
  각 checkpoint에 byte size/SHA-256을 기록하고 atomic rename으로 `resume.json`을 갱신합니다.
- 최근2세대와 각 세대가 참조하는 best student를 보존합니다. 현재 파일이 전송 중 손상된 경우
  체크섬이 맞는 직전 세대를 읽고 어느 세대로 재개했는지 기록합니다.
- method/config/code/data/초기 state/공식 소스·가중치/runtime 버전이 다르면 재개를 거부합니다.
  같은 출력에 새 학습을 덮어쓰지 않습니다. 다른 폴더로 옮긴 bundle에서의 재개도 지원합니다.
- H200 설치는 성공한 smoke의 NumPy2.2.6/Pillow12.3.0/SciPy1.15.3을 constraints로
  고정합니다. PyTorch2.11.0/torchvision0.26.0은 기존 프로젝트 의존성에 고정되어 있고,
  원본 timm0.4.12는 프로젝트 timm1.0.27과 분리합니다. 실제 runtime 버전도 저장합니다.
- `--max-hours`는 runner 준비 시작부터 세는 **soft limit**입니다. 도달 시 현재 batch 또는
  val 이미지 처리 후 저장합니다. shell의 설치·압축 해제 시간, 마지막 저장 시간은 별도입니다.
  승인 종료 시각보다 여유 있게 설정해야 하며 강제 종료를 막아 주는 기능은 아닙니다.
- SIGINT/SIGTERM을 받으면 가능한 경우 저장 후 `paused`로 종료합니다.
  SIGKILL·노드/컨테이너 강제 삭제 시에는 마지막으로 보존된 checkpoint까지만 복구할 수 있습니다.
- 재개 후 epoch 이력은 checkpoint의 이력에서 다시 작성합니다.
  `steps_*.jsonl`은 실행 시도별 원시 로그이므로 재처리된 step이 여러 파일에 존재할 수 있습니다.
  확정 이력은 `history.json`과 `resume.json`을 기준으로 읽습니다.

파일 기록·완전 검증에는 시간과 저장 공간이 추가로 필요합니다.
한 방법의 유지 checkpoint는 대략6–10GB이며 저장 중 임시 공간도 필요합니다.
데이터·런타임 cache 및 학습 출력을 위해 작업 공간에 여유 공간을 확보합니다.
실제 H200 본학습의 수렴·소요 시간·중단 후 CUDA 재개 오차는 아직 본 실행 결과가 없습니다.
