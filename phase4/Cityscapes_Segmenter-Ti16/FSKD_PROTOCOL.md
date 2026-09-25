# Segmenter-Tiny의 FSKD 이식 명세 v1

2026-09-26 사용자 결정: Tiny smoke에 **FSKD만 추가하고 C2VKD는 보류**한다.
기존 OpenMMLab DeepLabV3-R101-D8 teacher와 Tiny 초기 가중치를 유지한다.
CIRKD teacher, CLIP pool, C2VKD loss를 적재하거나 실행하지 않는다.

## 비교 해석

표기는 **FSKD* (DeiT-Ti recipe transfer)**다. 저자의 공개 분류 예제를 공통
Cityscapes segmentation 조건으로 이식하며, 저자 Cityscapes 설정의 완전 재현이나
최적 계수라는 주장은 하지 않는다. SegFormer-B0에서 사용한 FSKD*는 PiT-Ti의
다른 예제를 이식한 것이므로 이 설정과 혼합하지 않는다.

공식 저장소 확인 commit: `969dddf278b2c9f2dadde504326fa9d704c5a5aa`.

- [공식 README](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/README.md)의
  ImageNet/DeiT-Ti 예제는 stage 1·2, spatial `lg`, channel `cg`, global100,
  patch1, attention1,000,000, logit KD1을 명시한다.
- [실행 모델의 stage_info](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/custom_model/deit.py)는
  `deit_ti`의 stage 1·2를 **0-based block 0·3**으로 정의한다.
  `custom_forward/vision_transformer.py`의 별도 timm wrapper(stage1 index1)와 구분한다.
- [손실 코드](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/distillers/simi.py)의
  CLS attention, feature alignment, global/patch reduction과 soft-rank를 따른다.

## 고정값과 segmentation 보완

| 항목 | 설정 | 근거 |
|---|---|---|
| Teacher·student·데이터·optimizer | Tiny 공통 조건 그대로 | 우리 공정 비교 조건 |
| Feature 쌍 | Tiny block 0,3 → ResNet layer1,2 | 공개 실행 모델의 stage 선택과 index를 이식 |
| 공간 정렬 | Linear → GELU | 공개 `lg` |
| 채널 정렬 | Conv3×3 → GELU | 공개 `cg` |
| Attention | 마지막 encoder block의 CLS→patch, head 평균 | 공개 FSKD; Tiny는 실제 CLS가 존재 |
| Attention teacher | layer4를 32×32로 resize, 채널 제곱 평균 후 spatial softmax | 공개 FSKD |
| CE / logit KD / global / patch / attention | **1 / 1 / 100 / 1 / 1,000,000** | 공개 DeiT-Ti 예제·parser 기본값 이식 |
| logit KD | T1, KL(teacher‖student), 유효 픽셀 평균 | 분류 batchmean의 segmentation 이식 |
| ignore / logits resize | 255 제외, bilinear/align_corners=False | 현재 Tiny·OpenMMLab 좌표 조건 |
| soft-rank | torchsort0.1.10, L2 strength1 | 공개 호출·기본값 |
| β 후보 / 종료 controller | 없음, 위 고정 계수 사용 | 기존 LG/ALG/iBKD 탐색과 별도 |

512 crop에서 Tiny feature는 모두 `[B,192,32,32]`다. ResNet layer1은
`[B,256,128,128]`, layer2는 `[B,512,64,64]`다. Spatial Linear의 크기는
각각 1024→16384, 1024→4096이며, channel Conv는 192→256, 192→512다.
실제 feature shape가 이 조건과 다르면 중단한다.

원래 DeiT-Ti 분류 예제에는 CLS/DIST 토큰과 분류 head가 있다. 이번에는 공통
Segmenter-Tiny의 CLS 1개·segmentation decoder를 유지하고, logit KD는 픽셀 logits에
적용한다. DIST 토큰·추가 분류 head·원래 RegNet teacher를 도입하지 않는다.
이는 분류 예제 전체를 복제한 것이 아니라 공개 FSKD 손실·연결 선택의 segmentation 이식이다.

목적함수:

```text
L = CE + logit_KD + 100*global + patch + 1,000,000*attention
global = sum_stage(MSEmean(Align(S), T) / actual_batch_size)
patch  = sum_stage(MSEmean(cosine_Gram(S), cosine_Gram(resize(T))))
attention = mean_batch(6*sum((soft_rank(CLS_attention)-soft_rank(T_importance))²)/(N*(N²-1)))
```

Global의 추가 `/B`는 공개 코드대로 실제 batch8을 사용한다. Patch는 이미지별
모든 위치 쌍을 사용하고 정확한 chunk 계산을 재사용한다. Teacher만 detach하며
CLS attention은 실제 forward에서 읽어 student까지 gradient를 유지한다.
soft-rank를 hard argsort로 대체하거나 attention을 다시 forward하지 않는다.
모든 공간 feature 손실에는 GT ignore mask를 씌우지 않고 CE/logit KD에서만 제외한다.
순위 손실은 `1−Spearman`과 대수적으로 같은 식으로 계산해 FP32 상쇄를 줄인다.

Alignment와 Gram 연산은 검증된 B0 공통 primitive를 재사용하며 해당 소스 SHA도
Tiny 실행의 source hash에 포함한다. B0 모델·teacher·데이터·훈련 runner는 호출하지 않는다.
Adapter는 seed1001로 초기화하고 Tiny와 같은 SGD optimizer에 포함한다.
학습 RNG seed2001, 입력 및 student 초기 상태는 다른 다섯 방법과 동일해야 한다.

## Smoke에서 확인할 것

`smoke25_fskd_v2.json`은 기존 5경로에 FSKD* 하나를 더한 **총 6경로**다.
FSKD도 update 없이 25개 배치의 초기 구성 손실을 측정한 후 상태를 복원하고,
고정 계수로 25 update 및 마지막 update 재개 검사를 한다. CE 비율은 진단만 하며
이를 근거로 FSKD 계수를 자동 조정하지 않는다.

최종 JSON의 FSKD `beta=null`, `guidance_multiplier=1`은 고정 계수를 이미 적용한
손실을 뜻한다. 증류를 끈 것이 아니다. `components`와 `weighted_components`에
각 항의 원래 값과 가중값을 모두 남긴다. `soft_rank_execution`은 CUDA 확장의
forward/backward 검사이며, val2장 지표는 연결 진단이다.

이 smoke는 긴 학습의 안정성·최적성을 검증하지 않는다. 초기 score가 낮다는 이유로
비교군을 제거하지 않으며, 향후 계수/연결을 바꾸면 새 revision으로 분리한다.

## 로컬 사전 검사

2026-09-26 관련 단위 검사 41개를 통과했다. 공식 Tiny NPZ와 32×32 합성 입력을
사용한 CPU 검사에서 attention hook 추가 전후 logits가 완전히 같았으며, 실제
마지막 encoder QKV까지 FSKD attention gradient가 전달됨을 확인했다.
전체 고정 FSKD loss의 backward와 student·adapter·optimizer·RNG 복원 후
동일 update 재현도 통과했다. Teacher feature는 이 로컬 검사에서 합성값을 사용했다.
**실제 Cityscapes/H200 실행 및 25-step 안정성 결과는 아직 없다.**
