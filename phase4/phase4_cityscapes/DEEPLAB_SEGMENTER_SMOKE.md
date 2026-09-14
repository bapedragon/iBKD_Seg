# DeepLabV3 → Segmenter: 첫 GPU smoke

2026-09-14 사용자 요청: CNN → ViT segmentation에서 LG·ALG·iBKD를 비교하되
먼저 H200 smoke 이슈를 제출했습니다. 제출 당시 실제 Cityscapes 데이터가 준비되지 않아
해당 실행은 합성 입력과 임의 초기 가중치만 사용했습니다.
이후 사용자 제공 H200 로그에서 네 방법 통과를 확인했습니다.
실제 ZIP 다운로드 이후 작업은 [데이터 준비·H200 전달 안내](DATA_PREPARATION.md)에 기록합니다.

## 무엇이 이미 정해져 있는가

| 구분 | 정해진 내용과 이번 적용 |
|---|---|
| Cityscapes 공식 데이터·평가 | Fine train 2,975 / val 500 / test 1,525, 19클래스, void 제외. Test 정답 비공개. 공식 대표 평가는 mIoU이며 iIoU 등도 제공 |
| 공개 모델의 학습 설정 | 모델·논문마다 다름. Cityscapes 전체가 동일 optimizer, crop, batch, epoch를 강제하지 않음 |
| 이번 연구의 선택 | DeepLabV3 CNN teacher → Segmenter ViT student, Vanilla/LG/ALG/iBKD 비교, pixel accuracy 우선·같은 checkpoint의 mIoU 보조 |
| 이번 smoke | 구조·gradient·메모리·checkpoint 검증. 실제 train/val/test 또는 공개 pretrained checkpoint를 읽지 않음 |

Pixel accuracy는 유효 픽셀 전체에서 맞힌 비율입니다. Cityscapes의 공식 주 지표가
pixel accuracy라는 뜻은 아닙니다. 실제 학습에서는 val 정확도로 checkpoint를 선택하고
같은 checkpoint의 mIoU를 기록하는 사용자 지정 평가 방식을 사용할 예정입니다.

## 모델과 공개 구현의 관계

- Teacher: torchvision 0.26.0의 `deeplabv3_resnet101`, output stride 8,
  ASPP + 19클래스 classifier. **DeepLabV3**이며 V3+가 아닙니다.
  `weights=None`, `weights_backbone=None`, `aux_loss=False`로 다운로드를 막습니다.
  모든 guided 방법에서 동일 seed의 teacher 전체 state를 공유하고 `eval()`/freeze합니다.
- Student: `vit_small_patch16_384` 구조(12 blocks, 384채널, patch16) +
  2-layer mask Transformer decoder, 19개 class token. 모든 방법의 초기 encoder/decoder state가 같습니다.
- Segmenter mask decoder는 저자 공개 구현 commit
  `20d1bfad354165ee45c3f65972a4d9c131f58d53`의 구조를 현재 timm으로 이식했습니다.
  Block은 timm/SDPA, mask cosine 정규화는 FP32와 0벡터 epsilon을 사용합니다.
  [MIT 고지](SEGMENTER_LICENSE.txt)를 포함합니다.
- 저자 공개 Cityscapes model-zoo 수치는 **Seg-L-Mask/16**입니다.
  이번 Seg-S-Mask/16은 초기 계산량을 줄인 공개 Segmenter 계열 선택이며,
  저자의 Cityscapes 수치를 그대로 재현한다고 주장하지 않습니다.
- torchvision DeepLabV3와 MMSegmentation DeepLabV3는 checkpoint 형식과 세부 구현이 다릅니다.
  이번 코드는 MMSeg Cityscapes checkpoint를 그대로 읽는 변환기가 아닙니다.
  후속 본학습 전 사용할 공개 가중치·구현·전처리를 함께 고정하고 strict-load 및 teacher val 검증을 해야 합니다.

## 이번 실행에 고정한 조건

| 항목 | GPU smoke |
|---|---|
| 비교 | Vanilla / LG / ALG / iBKD λ=0.25, 순차 subprocess |
| 입력 | 모든 방법에 같은 합성 RGB·19클래스 mask, 앞 2행은 ignore255 |
| Crop / batch | 768×768 / 2, 임의 축소 fallback 없음 |
| 정규화 | 동일 RGB 공간 crop에서 teacher는 ImageNet mean/std, student는 mean/std 0.5 |
| 학습 점검 | SGD lr0.01, momentum0.9, wd0, BF16, gradient clip1, 각 3step |
| 재시작 점검 | 2step 뒤 저장 → 3번째 update → strict reload와 optimizer/RNG/controller 복원 → 같은 update 비교 |
| 평가 경로 | 합성 1024×2048, window768 / stride512, single-scale logits 평균, void 제외 |
| 출력 | 각 방법의 loss·feature shape·step 시간·GPU memory·합성 정확도/mIoU·checkpoint hash |

768 crop와 512 stride는 저자의 공개 Cityscapes 기본 설정을 참고했습니다.
그 공개 설정은 global batch8, 216epoch, lr0.01이며 기본 optimizer는 SGD입니다.
이번 batch2·3step·BF16·gradient clipping은 **smoke 전용 변경**입니다.
학습 스케줄 및 본실험 batch를 확정하거나 전체 소요 시간을 추정하는 실행이 아닙니다.

LG/ALG의 block `[0,6,11]` ↔ teacher layer2/3/4 대응과 더 큰 grid로 정렬하는 규칙을 유지합니다.
ViT-S에 맞춰 adapter 입력 채널만 192→384로 바꿉니다. iBKD는 12-block 집계,
deformable CBAM, 모든 key를 보는 cross-attention, λ0.25를 유지합니다.
Teacher 세 stage는 dilation 때문에 모두 stride8이며 768 입력에서 96×96입니다.
큰 attention은 query256 chunk + checkpoint로 계산하며 local attention으로 대체하지 않습니다.

β2.5, window50, threshold−0.02, ALG warm-up0과 iBKD warm-up20을 사용합니다.
실제 gradient smoke는 하나의 합성 epoch 3step으로 guidance 활성 경로를 확인합니다.
별도 고정 손실 controller 진단으로 ALG/iBKD의 서로 다른 종료 조건과 상태 복원을 확인합니다.
이 진단의 종료 epoch는 실제 Cityscapes guidance 종료 시점을 뜻하지 않습니다.

## 통과 조건과 결과 범위

1. 네 방법 모두 CE·guidance·gradient가 유한하고 encoder/decoder/guidance로 gradient가 흐름.
2. Frozen teacher의 gradient가 없고 parameter/BatchNorm buffer hash가 보존됨.
3. 네 방법의 초기 student/decoder hash와 세 guided 방법의 teacher hash가 각각 같음.
4. Checkpoint strict reload와 재개 update 일치. CPU는 exact, CUDA는 rtol2e−5/atol2e−6.
5. Native sliding-window가 void를 제외한 전체 픽셀을 평가하고 두 지표를 저장함.
6. `[CITYSCAPES_ARCH_SMOKE_DONE] status=passed methods=4/4 scientific_result=false` 출력.

실패하면 partial 결과와 실패 방법을 저장하고 중단합니다. 방법별로 batch/해상도를 자동 변경하지 않습니다.
Synthetic score로 방법 우열을 판단하지 않습니다. Random teacher의 feature loss 크기는
실제 학습된 teacher와 다르므로 손실 균형·수렴 여부도 이 smoke에서 결론 내리지 않습니다.

## 실제 데이터 준비 후 다음 단계

[Cityscapes 공식 다운로드](https://www.cityscapes-dataset.com/downloads/)에서 사용자 계정으로
`leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`을 준비합니다. 다운로드나 credential 전송은
이번 요청에 포함하지 않습니다. 원본 `labelIds`를 읽는 데이터 audit은 기존 Cityscapes 모듈에 있습니다.

이후 실제 데이터 audit, 공개 teacher/encoder 가중치 provenance와 전처리 검증,
DeepLabV3/Segmenter용 실제 학습 연결 및 짧은 real-data smoke를 거쳐 본실험 조건을 고정합니다.
현재 이전 `cityscapes.run train/matrix`는 자체 decoder pilot이므로 새 모델 본학습 명령으로 쓰지 않습니다.

## 출처

- [공식 split·파일 구조](https://github.com/mcordts/cityscapesScripts/blob/master/README.md)
- [공식 평가](https://www.cityscapes-dataset.com/benchmarks/)
- [torchvision DeepLabV3](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/segmentation/deeplabv3.py)
- [Segmenter 공개 모델](https://github.com/rstrudel/segmenter/tree/20d1bfad354165ee45c3f65972a4d9c131f58d53)
- [Segmenter 데이터별 기본 설정](https://github.com/rstrudel/segmenter/blob/20d1bfad354165ee45c3f65972a4d9c131f58d53/segm/config.yml)
- [Segmenter 학습 기본값](https://github.com/rstrudel/segmenter/blob/20d1bfad354165ee45c3f65972a4d9c131f58d53/segm/train.py)
