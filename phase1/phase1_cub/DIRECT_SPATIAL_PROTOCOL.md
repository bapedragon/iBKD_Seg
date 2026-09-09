# Phase 1 CUB 직접 공간정보 진단 프로토콜

상태: **v1 지표·smoke 계약 LOCK — smoke 결과 확인 전 고정**

이 프로토콜은 분류 정확도나 frozen segmentation mIoU만으로 공간정보 보존을
간접 추론하지 않고, 동일한 classification-best encoder에서 위치 정보를 직접
측정하기 위한 후속 진단입니다. 주 비교는 잠긴 CUB v3와 같은 batch 128이며,
batch 64 sensitivity는 섞지 않습니다.

현재 실행 가능한 smoke 입력은 H200 issue 727의 encoder seed 1 네 개입니다.
진행 중인 batch-128 seed 2·3 결과를 회수하고 checkpoint hash를 고정한 뒤 같은
정의를 그대로 본실험으로 확장합니다. Smoke 값으로 방법, lambda, 지표 정의,
학습률 또는 본실험 범위를 선택하지 않습니다.

Machine-readable 계약은
[`configs/cub200_r50_224_b128_direct_spatial_smoke_v1.json`](configs/cub200_r50_224_b128_direct_spatial_smoke_v1.json)에
있습니다. SHA-256은
`55ac0598c11a4065f3b1416022e8fbb3de35b21cad94780ace2e2036d5430bc6`입니다.

## 공통 입력

- CUB split: train `5,394` / validation `600` / official test `5,794`
- 입력: 결과와 무관한 direct-square `224×224`, bilinear antialias, ImageNet 정규화
- encoder: scratch DeiT-Tiny/16, classification validation-best checkpoint
- 비교: `LG`, `ALG-w20`, `iBKD λ=0.25`, `iBKD λ=0.5`
- 주 조건: student batch 128, encoder seed `[1,2,3]`
- encoder는 `eval()` 및 완전 freeze하며 strict load와 파일/state hash를 검사
- 독립 반복 단위는 encoder seed입니다. Probe seed는 encoder 안에 중첩합니다.

## 1. Visible-part localization probe — 주 직접 지표

CUB가 제공하는 15개 part landmark 중 `visible=1`인 점만 사용합니다. 보이지 않는
part는 loss와 metric에서 모두 제외하고, 사후 누락 기준은 만들지 않습니다.

- feature: DeiT block 11의 final norm 전 patch feature `192×14×14`
- head: `Conv2d(192,15,1,bias=True)`, 총 `2,895` parameter
- target: part마다 `14×14` 2-D Gaussian heatmap, `σ=1` grid pixel, peak `1`
- loss: visible part에만 적용한 heatmap MSE
- 좌표: `x_grid=x×14/width-0.5`, `y_grid=y×14/height-0.5`; decode는 argmax
  patch 중심을 역변환하며 공식 floating coordinate를 반올림하거나 원점 이동하지 않음
- optimizer: SGD, momentum `0.9`, weight decay `0`, Nesterov 없음
- LR 후보: `[0.01, 0.03, 0.1]`, cosine to 0, `100` epoch, batch `64`
- probe seed: `[1,2,3,4,5]`; 같은 seed는 encoder마다 같은 head 초기값과 batch 순서
- validation 선택: visible-point micro PCK@0.1 최대, 동률이면 낮은 LR·이른 epoch
- 주 평가: `distance ≤ 0.1 × max(GT bbox width, GT bbox height)`인 visible point 비율
- 보조 평가: part별 PCK@0.1, bbox 최대 변 길이로 정규화한 평균 위치 오차

Gaussian heatmap과 MSE는 keypoint localization에서 널리 쓰이는 형태이고, grid에서
`σ=1`을 쓰는 공개 관행을 따릅니다. PCK는 CUB를 포함한 landmark 연구에서 쓰이는
bbox 최대 변 정규화를 명시적으로 채택했습니다. 근거는
[Multi-Scale Structure-Aware Network (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Lipeng_Ke_Multi-Scale_Structure-Aware_Network_ECCV_2018_paper.pdf)과
[Learning Articulated Shape With Keypoint Pseudo-Labels (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Stathopoulos_Learning_Articulated_Shape_With_Keypoint_Pseudo-Labels_From_Web_Images_CVPR_2023_paper.pdf)입니다.

Part probe의 official test는 모든 encoder·probe seed의 validation 선택을 먼저
완료한 뒤 엽니다. 선택된 probe만 test에 한 번 적용하며 test로 LR, epoch, 방법,
lambda를 고르지 않습니다.

## 2. Spatial linear CKA — 학습 없는 보조 지표

- 평가 표본: 고정 validation 600장만 사용하며 official test는 사용하지 않음
- teacher: issue 722 scratch ResNet-50의 `layer3`, `1024×14×14`
- student: DeiT block `0..11`의 final norm 전 patch token, 각각 `192×14×14`
- observation: 같은 이미지의 같은 `14×14` 공간 위치 하나
- metric: 전체 `600×196` observation을 중심화한 linear CKA
- 수치 계산: float64 streaming sufficient statistics
- 출력: 방법·encoder seed마다 12개 값과 layerwise heatmap

CKA는 채널 수가 다른 CNN과 ViT 표현도 비교할 수 있다는 장점 때문에 사용합니다.
다만 값이 높다는 사실은 teacher 표현 정렬을 뜻할 뿐, 그 자체로 유용한 공간정보가
더 많다는 결론은 아닙니다. 그래서 part probe의 보조 증거로만 해석합니다. 정의는
[Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)을
따릅니다.

## 3. Attention–GT localization — 학습 없는 보조 지표

- 12개 layer 각각에서 head 평균
- residual을 반영하도록 identity를 더하고 row normalization
- 이 행렬을 앞 layer부터 누적 곱해 최종 CLS-to-patch rollout `14×14` 생성
- foreground: 공식 segmentation mask의 grayscale 값 `>0`
- 주 지표: occupancy `≥0.5`인 patch를 positive로 한 global micro patch AP
- 보조 지표: 224 mask 내부의 attention peak 비율(pointing game), fractional
  foreground occupancy에 놓인 attention mass
- threshold tuning 없음
- 본실험은 고정 official test를 한 번 평가
- 정성 표본은 결과 확인 전 정한 test image ID
  `[787,2285,3735,5205,6691,8139,9597,11064]`

Rollout은 residual 연결과 여러 층을 함께 고려하는
[Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/)의
attention rollout 정의를 따릅니다. Attention은 모델 설명과 동일하지 않고 높은
localization 값도 인과 설명을 보장하지 않으므로 역시 보조 지표입니다.

## Smoke 범위

```bash
bash phase1/phase1_cub/scripts/run_r50_224_direct_spatial_smoke_b128_seed1.sh
```

Smoke는 issue 722 Teacher와 issue 727의 batch-128 seed-1 네 encoder를 GitHub
Release에서 내려받아 byte/state hash와 strict load를 검사합니다. Train과
validation에서 클래스별 가장 작은 image ID 한 장씩, 각각 200장을 사용합니다.

- part probe: 4 encoder × LR 3개 × probe seed 1 × 2 epoch = 후보 12개
- spatial CKA: 4 encoder × 12 block = 48개 값
- attention–GT: 4개 metric row
- 정성 결과: smoke validation에서 가장 작은 image ID 4장 × 4방법 = PNG 16개
- official test image/mask decode와 평가는 `0`회
- 요청 자원: H200 MIG slice `1`개, FP32 모델·float64 CKA accumulator

마지막 성공 marker는 다음과 같습니다.

```text
[DIRECT_SPATIAL_SMOKE_DONE] status=pass strict_loads=4 part_candidates=12 part_selections=4 cka_values=48 attention_rows=4 qualitative_pngs=16 official_test=0 ...
```

Smoke의 PCK·CKA·AP 수치는 코드와 좌표계가 실행된다는 진단값일 뿐 논문 결과가
아닙니다. 본실험 config는 진행 중인 seed 2·3 checkpoint의 파일·state SHA-256을
회수한 뒤 생성하며, 이 문서의 지표·split·학습·test 규칙은 변경하지 않습니다.
