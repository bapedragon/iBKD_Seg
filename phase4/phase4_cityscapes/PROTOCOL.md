# Cityscapes 직접 segmentation pilot v2 — 정확도 우선

이 문서는 이전 ResNet50/DeiT + 자체 decoder pilot 이력입니다.
현재 사용자가 선택한 DeepLabV3 → Segmenter smoke는
[별도 조건](DEEPLAB_SEGMENTER_SMOKE.md)을 따릅니다.

## 실험 목적과 위치

2026-09-14 사용자 요청에 따라 LG·ALG·iBKD의 Cityscapes 직접 학습 구성을 추가합니다.
기존 Pet/CUB의 분류→frozen probe와 다른 실험입니다. Phase 1의 No-Go 판단이나
후속 단계 진입 결정을 변경하지 않으며, iBKD의 우위를 전제로 하지 않습니다.
표준 다중 클래스 segmentation에 해당하는 별도 `phase4/phase4_cityscapes/` pilot입니다.

실제 Cityscapes 결과를 아직 관측하지 않았습니다. 사용자 요청에 따라 v2는 pixel accuracy를
주 지표로 변경하며, v1의 mIoU 선택 설정은 별도 파일로 보존합니다. Crop·batch·학습률·기간 등은 실행 가능한
출발점으로 정한 값이며 Cityscapes 최적 설정이라고 주장하지 않습니다. 이 설정으로
실험한 후 수정한다면 새 protocol ID·설정 파일·출력 폴더로 구분하고 비교 대상 전체를
다시 실행합니다. 기존 분류 논문 수치나 frozen probe mIoU와 같은 표에서 직접 비교하지 않습니다.

## 데이터와 공통 조건

- 공식 fine train 2,975장과 val 500장을 사용합니다. Coarse, train_extra, test는 사용하지 않습니다.
- 원본 `*_gtFine_labelIds.png`를 다음 순서로 trainId 0–18에 매핑합니다.
  `[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]`.
- 나머지는 255(ignore)로 처리하며 pixel CE와 confusion matrix에서 제외합니다.
- 학습은 원본 해상도에 0.5–2.0 배율의 bilinear resize → 512×512 crop → 확률 0.5
  좌우 반전을 적용합니다. 마스크 resize는 nearest입니다. 이미지와 마스크는 같은 변환을 공유합니다.
- 단일 클래스 비율 0.75 미만의 crop을 최대 10번 탐색하고, 실패해도 마지막 crop을
  유지합니다. 표본을 버리지 않습니다. Batch 전체가 void인 경우 optimizer update를
  건너뛰고 `skipped_void_samples`에 기록합니다. 모든 batch가 void면 실패합니다.
- RGB 정규화 mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`를 공통 사용합니다.
  이 정규화 상수는 ImageNet 사전학습 사용을 의미하지 않습니다.
- 같은 seed의 방법들은 같은 student+decoder 초기 가중치, 이미지 순서, crop/flip을 공유합니다.
  변환 난수는 `(seed, epoch, image_id)`에서 만들므로 worker 수나 방법에 의존하지 않습니다.
- Teacher 입력도 student와 **같은 crop·해상도**입니다. 기존 분류의 32×32 teacher 입력을 사용하지 않습니다.

## 모델과 손실

Teacher는 scratch ResNet-50의 layer1–4를 segmentation decoder에 연결합니다.
Guidance에는 layer2–4의 512/1,024/2,048채널 feature를 사용합니다. Teacher를 100 epoch
학습한 뒤 val pixel accuracy로 선택한 `best.pt`를 모든 guided student에 공유하며, 이후 teacher는
`eval()`과 `requires_grad_(False)`로 고정합니다. 분류용 teacher checkpoint를 받지 않습니다.

Student는 scratch DeiT-Tiny/16입니다. 12개 block의 pre-norm spatial feature를 추출하고,
decoder에는 block `[2,5,8,11]`을 전달합니다. CLS token은 공간 map에서 제거합니다.
DeiT의 positional embedding을 crop grid 크기로 초기화하고 rectangular 입력을 지원합니다.

Decoder는 네 feature를 각각 1×1 conv→GroupNorm→GELU로 128채널에 투영하고,
첫 feature 해상도로 resize해 concatenate한 뒤 3×3 conv→GroupNorm→GELU→19채널
classifier를 적용합니다. Teacher와 student의 입력 채널만 다르고 모든 student는 정확히
같은 decoder를 사용합니다. 학습 CE는 logits를 crop 해상도로 bilinear upsample해서 계산합니다.
기성 UPerNet/DeepLab의 수치를 재현하는 구현은 아닙니다.

| 방법 | 목적함수와 guidance 일정 |
|---|---|
| Vanilla | Pixel CE |
| LG | CE + 2.5 × LG, 전체 기간 활성 |
| ALG | CE + β × LG, 원래 ALG controller로 β=2.5→0 |
| iBKD | CE + β × (0.75 × alignment + 0.25 × fusion), 기존 iBKD controller로 β=2.5→0 |

- LG/ALG는 student block `[0,6,11]`과 teacher 세 단계를 대응시킵니다.
- 기존 공통 구현의 1×1 projection, 더 큰 feature grid로 정렬, stage별 mean MSE의 합을 재사용합니다.
- iBKD는 기존 V1의 12-layer 집계, deformable channel/spatial attention, cross-attention을 재사용합니다.
- LG/ALG/iBKD의 feature loss는 feature map 전체에 적용합니다. Void 제외는 pixel CE와
  평가에 적용하며, feature guidance에 새 마스크 정책을 추가하지 않습니다.
- Attention은 각 query가 **전체 key/value**를 보는 연산을 유지합니다. Query 256개씩
  SDPA를 호출하고 activation checkpointing으로 재계산합니다. Window/local attention이나
  grid pooling으로 바꾸지 않았습니다. 출력과 gradient의 기존 구현 대비 일치를 테스트합니다.
- ALG: window 50, threshold −0.02, `>=` 종료, controller warm-up 0 epoch.
- iBKD: 기존 iBKD 평균 정규화, window 50, threshold −0.02, `>` 종료, controller warm-up 20 epoch.
- ALG 조기 종료가 생겨도 해당 실행 결과를 대체하지 않습니다. warm-up 변경 진단은
  별도 config/ID로 표시해야 합니다. 이식된 조건이므로 원 논문의 Cityscapes 공식 구현이라 부르지 않습니다.

## 학습·선택·보고

공통 AdamW lr `5e-4`, weight decay `0.05`, batch 4, 100 epochs입니다. Bias, 1차원
파라미터, positional embedding, CLS token에는 weight decay를 적용하지 않습니다.
LR은 처음 5 epoch에서 0.001배부터 선형 warm-up하고, 이후 power 0.9 polynomial decay를 사용합니다.
Gradient norm은 1.0으로 제한하며 BF16 autocast를 사용합니다. BF16에는 GradScaler를 사용하지 않습니다.
Student drop path 0.1, student block과 iBKD attention의 activation checkpointing을 활성화합니다.

Val은 원본 1024×2048에서 512×512 crop, stride 256의 sliding window로 평가합니다.
겹치는 영역은 float32 logits를 평균하고 마지막에 argmax합니다. Multi-scale/flip TTA는 없습니다.
평가 전체의 하나의 19×19 confusion matrix에서 class IoU와 mIoU를 구합니다.
평가 대상 클래스가 GT·예측 모두에 없으면 IoU는 null로 표시하며 평균에서 제외합니다.
실제 500장 val에서는 19클래스 평가가 아니면 오류로 중단합니다.

매 5 epoch 및 마지막 epoch에서 평가하고 가장 높은 val pixel accuracy의 checkpoint를 선택합니다.
Pixel accuracy는 전체 유효 픽셀 중 정답을 맞힌 비율이며, ignore 픽셀은 제외합니다.
mIoU와 class IoU는 그 동일한 checkpoint에서 보고합니다.
동률이면 앞선 epoch를 유지합니다. 별도 test는 사용하지 않으므로 결과는 명확히
`accuracy-selected validation pixel accuracy / mIoU`로 보고합니다. 3개 student seed는 같은 teacher를 공유하므로
seed SD는 student 학습 변동이며 teacher 재학습 변동까지 포함하지 않습니다.

논문의 data-scarce 주장에 직접 연결하려면 별도 후속 protocol로 train fraction 및
teacher의 라벨 접근 범위를 함께 고정해야 합니다. 현재 v2는 공식 train 전체를 사용합니다.

## 재현성과 실행 한계

Manifest에는 모든 입력 파일의 byte size·SHA-256과 순서를 보관합니다. 실행은 config,
manifest, 사용 Python source, 초기 model state, teacher checkpoint의 해시를 저장합니다.
Resume은 이 계약이 일치할 때만 허용하며 epoch 단위로 optimizer/controller/torch RNG를 복원합니다.
같은 CPU 환경의 중단/재개 수치 일치를 검증합니다. 다른 CUDA 환경 사이의 bitwise
재현성은 보장하지 않습니다. Deformable convolution과 fused attention의 CUDA 연산은
하드웨어·backend에 따른 차이가 있을 수 있습니다.

`preflight`는 실제 설정의 tensor shape와 optimizer를 사용하는 합성 실행입니다.
실제 데이터 I/O, 전체 validation, 학습 수렴 시간을 포함하지 않습니다. 현재 로컬은 CUDA가
없으므로 512×512·batch4·BF16의 H200 동작/메모리는 서버에서 이 명령으로 확인해야 합니다.
상용 서비스 실행, GPU 작업 제출, 데이터 다운로드는 이 구성 작업에 포함하지 않았습니다.

## 근거

- [Cityscapes 공식 파일 구조·split](https://github.com/mcordts/cityscapesScripts/blob/master/README.md)
- [공식 labelId/trainId 정의](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)
- [Cityscapes 공식 평가 기준](https://www.cityscapes-dataset.com/benchmarks/)
- [Locality Guidance](https://arxiv.org/abs/2207.10026), [ALG 논문](https://doi.org/10.1109/TNNLS.2024.3515076)
- [공유 guidance 구현](../../src/ibkd_seg/phase1/models.py), [controller 구현](../../src/ibkd_seg/phase1/controllers.py)
