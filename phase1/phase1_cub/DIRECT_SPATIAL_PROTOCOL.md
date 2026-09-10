# Phase 1 CUB 직접 공간정보 진단 프로토콜

상태: **v2 본실험 계약 LOCK — seed 1 실행·감사 완료, seed 2·3 smoke 통과·본실험 미실행**

이 프로토콜은 분류 정확도나 frozen segmentation mIoU만으로 공간정보 보존을
간접 추론하지 않고, 동일한 classification-best encoder에서 위치 정보를 직접
측정하기 위한 후속 진단입니다. 주 비교는 잠긴 CUB v3와 같은 batch 128이며,
batch 64 sensitivity는 섞지 않습니다.

H200 issue 727의 encoder seed 1 네 개로 smoke 실행 계약을 통과했습니다. 첫 v1
본실험은 어떤 probe도 학습하거나 official test를 열기 전에 CUB 원본 annotation
예외를 발견하고 중단됐습니다. 이미지 ID `5007`은 크기가 `500×333`인데 공식
visible인 part 4 좌표가 `(405,344)`입니다. v2는 이를 포함한 out-of-frame visible
part를 모든 방법에서 같은 규칙으로 제외하고 좌표를 clipping하지 않습니다. 이
변경은 방법별 결과를 보기 전에 이루어졌으며 방법, lambda, 지표, 학습률과 본실험
범위는 바꾸지 않았습니다.

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

CUB가 제공하는 15개 part landmark 중 `visible=1`이면서 원본 이미지의 연속 좌표
영역 `0≤x≤width`, `0≤y≤height` 안에 있는 점만 valid로 사용합니다. 보이지 않거나
이미지 밖인 part는 loss, validation 선택과 PCK에서 모두 제외합니다. 이미지 밖
좌표를 경계로 clipping하거나 다른 위치로 바꾸지 않습니다.

Train/validation은 probe 학습 전에 전수 감사하고 official test는 20개 validation
선택을 모두 끝낸 뒤 처음 열어 별도로 감사합니다. Split별 공식 visible 수, valid
수, 제외된 part 수와 해당 image ID·part ID·원 좌표·이미지 크기를
`part_probe/annotation_validity_audit.json`과 로그에 남깁니다. Valid part가 하나도
없는 이미지가 발견되면 결과를 만들지 않고 중단합니다.

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
아닙니다. 이 문서의 지표·split·학습·test 규칙은 본실험에서 변경하지 않습니다.

## Seed 1 본실험 shard

```bash
bash phase1/phase1_cub/scripts/run_r50_224_direct_spatial_full_b128_seed1.sh
```

Machine-readable 실행 계약은
[`configs/cub200_r50_224_b128_seed1_direct_spatial_full_v2.json`](configs/cub200_r50_224_b128_seed1_direct_spatial_full_v2.json)에
있으며 SHA-256은
`90f7dc92b7e1ad27b6a4a4b68e91bb5e72ea021389304d87dda5950fac8e6017`입니다.

대체된 v1의 config는 실패 재현 기록으로 보존합니다. v1은 annotation을 읽는
단계에서 중단되어 part probe·CKA·attention 결과와 official-test 접근이 모두
`0`이므로 v2 결과와 섞이는 pilot 결과가 없습니다.

- encoder: batch-128 seed 1의 `LG`, `ALG-w20`, `iBKD λ=0.25`, `iBKD λ=0.5`
- part probe: 4 encoder × probe seed 5개 × LR 3개 × 100 epoch = 후보 60개
- validation 선택: 20개를 모두 완료하고 marker를 쓴 뒤 official test를 처음 엶
- part test: validation-selected probe 20개를 각각 한 번 평가
- spatial CKA: validation 600장 × 4 encoder × 12 block = 48개 값
- attention–GT: official test 5,794장 × 4 encoder = 4개 row
- 정성 결과: 고정 test image ID 8장 × 4 encoder = PNG 32개
- official test evaluation count: part 20 + attention 4 = 24
- 요청 자원: H200 MIG slice `1`개

마지막 성공 marker는 다음과 같습니다.

```text
[DIRECT_SPATIAL_FULL_SEED1_DONE] status=complete encoder_seed=1 strict_loads=4 part_candidates=60 part_selections=20 part_test=20 cka_values=48 attention_rows=4 qualitative_pngs=32 official_test=24 final_encoder_seed_inference=false excluded_oob_train=... excluded_oob_validation=... excluded_oob_test=... ...
```

이 shard는 protocol에 맞는 scientific result이지만 독립 encoder seed가 하나이므로
encoder-seed 표준편차나 최종 방법 우위는 확정하지 않습니다. Seed 2·3을 같은 설정으로
추가한 뒤 세 encoder seed의 평균과 표본 표준편차로 최종 해석합니다.

H200 issue 737에서 완료 gate를 모두 충족하고 20개 part-probe checkpoint를 독립
감사했습니다. 결과와 32개 정성 이미지는
[seed-1 결과 보고서](reports/direct_spatial/resnet50_224_b128_seed1_v2/RESULTS.md)에
고정했습니다. 주 Part PCK와 CKA block11은 LG가 가장 높았고 attention 지표는
엇갈렸으므로, seed 1은 iBKD의 전반적 공간정보 우위를 지지하지 않습니다.

## Seed 2·3 실행 전 smoke

```bash
bash phase1/phase1_cub/scripts/run_r50_224_direct_spatial_smoke_b128_seeds2_3.sh
```

Machine-readable 계약은
[`configs/cub200_r50_224_b128_seed2_3_direct_spatial_smoke_v2.json`](configs/cub200_r50_224_b128_seed2_3_direct_spatial_smoke_v2.json)에
있으며 SHA-256은
`bd71b02ebcca3240c7278819b4a34416a131914bd442887afcce6251a7e366d1`입니다.

Issue 730 Release의 batch-128 seed 2·3 encoder 8개와 issue 722 Teacher를
byte/state hash로 감사하고 strict-load합니다. 진단 정의는 seed-1 v2에서 바꾸지
않고, 실행·메모리·출력 계약만 다음 축소 범위에서 확인합니다.

- part probe: 8 encoder × LR 3개 × probe seed 1 × 2 epoch = 후보 24개
- validation 선택: encoder마다 1개, 총 8개
- spatial CKA: 8 encoder × 12 block = 96개 값
- attention–GT: validation subset의 8개 metric row
- 정성 결과: encoder별 고정 validation 4장, 총 PNG 32개
- official test image/mask decode와 평가: `0`회
- 요청 자원: H200 MIG slice `1`개

마지막 성공 marker는 다음과 같습니다.

```text
[DIRECT_SPATIAL_SMOKE_DONE] status=pass encoder_seeds=2,3 strict_loads=8 part_candidates=24 part_selections=8 cka_values=96 attention_rows=8 qualitative_pngs=32 official_test=0 ...
```

이 smoke 값은 논문 결과가 아니며 seed-1 결과 재확인, 방법·lambda 선택 또는
프로토콜 변경에 사용하지 않습니다. 통과 뒤 seed 2·3 본실험은 seed-1과 같은
probe seed 5개, LR 3개, 100 epoch 및 validation-before-test 규칙을 사용합니다.

H200 issue 738에서 `170.43`초, peak allocated `405,430,784` bytes, peak reserved
`501,219,328` bytes로 통과했습니다. Strict-load `8`, part 후보 `24`, 선택 `8`,
CKA `96`, attention row `8`, 정성 PNG `32`, official test `0`을 모두 충족했습니다.
이 수치의 방법별 순위는 본실험 설정이나 실행 여부를 바꾸는 데 사용하지 않았습니다.

## Seed 2·3 본실험 shard

```bash
bash phase1/phase1_cub/scripts/run_r50_224_direct_spatial_full_b128_seeds2_3.sh
```

Machine-readable 계약은
[`configs/cub200_r50_224_b128_seed2_3_direct_spatial_full_v2.json`](configs/cub200_r50_224_b128_seed2_3_direct_spatial_full_v2.json)에
있으며 SHA-256은
`54980771cf910543a3aba24c0a5ff86de6a0dce34866c662025409ab3abd0691`입니다.

- encoder: batch-128 seed 2·3의 네 방법, 총 8개
- part probe: 8 encoder × probe seed 5개 × LR 3개 × 100 epoch = 후보 120개
- validation 선택: 40개를 모두 완료하고 marker를 쓴 뒤 official test를 처음 엶
- part test: validation-selected probe 40개를 각각 한 번 평가
- spatial CKA: validation 600장 × 8 encoder × 12 block = 96개 값
- attention–GT: official test 5,794장 × 8 encoder = 8개 row
- 정성 결과: 고정 test image ID 8장 × 8 encoder = PNG 64개
- official test evaluation count: part 40 + attention 8 = 48
- 요청 자원: H200 MIG slice `1`개

마지막 성공 marker는 다음과 같습니다.

```text
[DIRECT_SPATIAL_FULL_SEED23_DONE] status=complete encoder_seeds=2,3 strict_loads=8 part_candidates=120 part_selections=40 part_test=40 cka_values=96 attention_rows=8 qualitative_pngs=64 official_test=48 final_encoder_seed_inference=false ...
```

Issue 737 seed-1 실행이 `1,232.68`초였으므로 동일 장비의 순수 본실험 시간은 약
`2,465`초, 즉 약 41분으로 예상합니다. 최초 Release·CUB 다운로드와 설치를 포함하면
약 45~60분을 잡으면 충분하며 10시간 제한과는 큰 차이가 있습니다. 이 shard 결과를
회수·감사한 뒤 issue 737 seed 1과 결합해 encoder seed `[1,2,3]` 최종 평균과 표본
표준편차를 계산합니다.
