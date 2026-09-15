# main-L0 기존 checkpoint aggregation 감사

상태: **PASS — 6/6 checkpoint byte hash·metadata·가중치 유한값 확인**

## 입력

- 계보: `main_l0_v3`만 사용
- issue 727: batch-128 encoder seed 1
- issue 730: batch-128 encoder seed 2·3
- 방법: iBKD λ=0.25, λ=0.5
- 총 checkpoint: 6개

Loader pilot에서 다시 학습한 `pilot-L0` checkpoint나 선택 L2 결과는 포함하지
않았습니다.

## 3-seed 평균

균등 혼합이면 각 block 확률은 `0.08333`, early/middle/late 질량은 각각
`0.33333`, 정규화 entropy는 `1.0`입니다.

| λ | teacher feature | 가장 큰 block 확률 | 정규화 entropy | early 0–3 | middle 4–7 | late 8–11 |
|---:|---|---:|---:|---:|---:|---:|
| 0.25 | layer2 | block 9 / 0.08803 | 0.99973 | 0.33447 | 0.32175 | 0.34378 |
| 0.25 | layer3 | block 11 / 0.10387 | 0.99499 | 0.29070 | 0.30685 | 0.40245 |
| 0.25 | layer4 | block 0 / 0.08766 | 0.99983 | 0.34229 | 0.32577 | 0.33194 |
| 0.5 | layer2 | block 1 / 0.08941 | 0.99930 | 0.35505 | 0.33513 | 0.30982 |
| 0.5 | layer3 | block 11 / 0.08709 | 0.99977 | 0.32263 | 0.33157 | 0.34580 |
| 0.5 | layer4 | block 0 / 0.08900 | 0.99965 | 0.34963 | 0.32858 | 0.32179 |

## 해석

최종 aggregation은 특정 block 하나를 강하게 선택하지 않았고 대부분 균등 평균에
가깝습니다. 가장 뚜렷한 패턴은 λ=0.25가 teacher layer3에 대응할 때 후반 block
8–11에 약 `40.25%`를 둔 것입니다. 그래도 top block 하나의 평균 확률은
`10.39%`라서 날카로운 선택이라고 보기는 어렵습니다.

따라서 현재 관찰만으로 “iBKD가 특정 유리한 레이어를 강하게 골라 분류 성능을
올렸다”고 설명하기는 어렵습니다. 다만 최종 가중치가 균등에 가깝다는 사실은 학습
도중의 gradient 경로가 효과가 없었다는 뜻은 아닙니다. 다음 사전 고정 실험은
canonical `learned_all`을 `fixed_uniform_all`, `fixed_stage_match`, `fixed_last`와
동일 seed에서 다시 학습해 이 가능성을 인과적으로 분리합니다.

세부 checkpoint별 18개 row는 `aggregation_per_checkpoint_stage.csv`, raw logit과
확률 및 hash는 `aggregation_checkpoint_audit.json`에 보존합니다.
