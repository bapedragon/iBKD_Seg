# main-L0 iBKD 레이어 연결 원인 분석 프로토콜

상태: **기존 checkpoint 관찰 감사 완료, 인과 full 설정 사전 고정, smoke 대기**

## 질문

완료된 `main_l0_v3`에서 iBKD λ=0.25는 guided 네 방법 중 평균 분류 정확도가 가장
높았지만 마지막 block 공간 probe는 LG보다 낮았습니다. 확인할 질문은 다음입니다.

> iBKD의 분류 이득은 마지막 block 자체가 더 공간적이어서가 아니라, 학습 중 여러
> student block을 teacher stage에 연결한 경로가 student 전체의 최적화를 바꿨기
> 때문인가?

iBKD의 aggregation·projection·fusion module은 학습에만 사용되고 추론 때
제거됩니다. 따라서 가능한 효과는 추가 추론 용량이 아니라 guidance gradient가
student 가중치를 바꾼 결과입니다.

## M0 — 기존 checkpoint 관찰 분석

- 입력: `main_l0_v3`, batch 128, issue 727 seed 1 + issue 730 seed 2·3
- 대상: iBKD λ=0.25와 λ=0.5, 총 6개 validation-selected checkpoint
- 측정: teacher stage별 12-block softmax 확률, top block, 정규화 entropy,
  early/middle/late block mass
- 역할: 무엇이 학습됐는지 관찰하는 진단이며 인과 증거가 아님

[감사 결과](reports/main_l0_aggregation_checkpoint_audit_v1/RESULTS.md)에서 모든
row의 정규화 entropy가 `0.99499` 이상이었습니다. 즉 최종 연결은 전반적으로
`1/12` 균등 혼합과 가깝습니다. λ=0.25의 teacher `layer3` row만 후반 block
`8–11` 질량이 3-seed 평균 `0.40245`로 가장 뚜렷하게 기울었습니다.

이 결과만으로 학습 중 aggregation gradient의 효과가 없다고 결론낼 수 없습니다.
그래서 다음 M1에서 연결 방식만 바꾸어 다시 학습합니다.

## M1 — 레이어 연결 인과 ablation

| ID | block→teacher-stage 연결 | 확인하는 것 |
|---|---|---|
| `learned_all` | stage마다 12개 block softmax 가중치를 학습 | 동일 실행 안에서 다시 학습하는 canonical 대조군 |
| `fixed_uniform_all` | 세 stage 모두 12개 block 산술평균 | 가중치 **학습** 자체가 필요한가 |
| `fixed_stage_match` | stage별 block `0/6/11` one-hot | LG식 고정 stage 대응보다 learned cross-layer 연결이 나은가 |
| `fixed_last` | 세 stage 모두 block 11 one-hot | 마지막 block만 연결해도 분류 효과가 남는가 |

네 조건 모두 iBKD λ=0.25입니다. Canonical checkpoint를 단순 재사용하지 않고 같은
실행 계열에서 `learned_all`도 다시 학습해 코드·환경 차이를 paired 비교에서
제거합니다.

### 바꾸지 않는 값

- CUB split `5,394 / 600 / 5,794`, split seed `2027`
- `main-L0` loader `l0_current_strong`
- issue 722 scratch ResNet-50/224 teacher 한 개
- DeiT-Tiny/16, batch 128, encoder seeds `[1,2,3]`, 300 epoch
- AdamW `5e-4`, weight decay `0.05`, 20-epoch LR warm-up, cosine schedule
- iBKD λ `0.25`, beta `2.5`, controller warm-up `20`, threshold `-0.02`, window `50`
- classification checkpoint는 validation macro top-1 최대, 동률이면 이른 epoch
- 모든 방법에 같은 final-block frozen binary segmentation probe
- probe seeds `[1,2,3,4,5]`, LR `[0.01,0.03,0.1]`, 100 epoch

Controller 종료 epoch는 고정 입력이 아니라 연결 변경 뒤에 생기는 **측정 대상**으로
기록합니다. 종료 epoch가 조건마다 크게 달라 결과 해석을 지배하면, 이 protocol을
바꾸지 않고 별도의 matched-guidance-duration 실험을 새 버전으로 추가합니다.

### 평가와 해석

- 주 원인 지표: official-test classification macro top-1의 seed별 paired 차이
- 보조 지표: frozen segmentation probe input-224 mIoU, controller 종료 epoch,
  aggregation 확률
- `learned_all`이 `fixed_uniform_all`과 `fixed_stage_match`보다 seed별로 일관되게
  높으면 learned connection 설명을 지지합니다.
- `fixed_uniform_all`이 사실상 같으면 learned **가중치**가 핵심 원인이라는 설명은
  약해집니다. 이때 projection/fusion loss나 단순 다층 gradient 경로를 다음 후보로
  봅니다.
- `fixed_last`도 같으면 여러 층 연결 자체보다 마지막 층 guidance가 원인일 가능성이
  커집니다.
- 어느 결과도 완료된 `main_l0_v3` 주 결과를 대체하지 않습니다.

공식 full 계약은
`configs/cub200_r50_224_b128_main_l0_ibkd_connection_full_v1.json`, SHA-256
`1650e76da40c235292fca166b8058c1a9ee279165a275b59122f505f3b7235d8`에
고정했습니다.

## Smoke

Smoke는 encoder seed 1에서 네 연결을 각 2 epoch 학습하고, 각 frozen encoder에
probe seed 1 × LR 3개 × 2 epoch를 적용합니다. Train/validation 전체를 사용해
시간과 메모리를 현실적으로 측정하지만 official test dataset은 생성하지 않습니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_connection_smoke_b128_seed1.sh
```

Smoke config SHA-256은
`1fb71d3e076e592df3209c176bcea05a296f4745b243664b09afccef59de2400`입니다.
완료 기준은 분류/checkpoint/aggregation audit `4/4`, probe 후보 `12/12`,
validation 선택/checkpoint `4/4`, official-test 평가 `0`입니다. Smoke 결과로
variant, λ, seed, LR, epoch 또는 full 수행 여부를 선택하지 않습니다. Full job
분할만 실제 시간으로 정합니다.
