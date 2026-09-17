# main-L0 iBKD 레이어 연결 원인 분석 프로토콜

상태: **기존 checkpoint 관찰 감사 완료, 본실험 과학 설정·실행 분할 고정**

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

## M1 재개 전 gate — issue 760 `learned_all` 단일 재현

기존 M1 seed-1의 `learned_all`은 test macro Top-1 `10.4096%`로 main-L0의
`27.0361%`를 재현하지 못했습니다. 이후 제어 A/A 두 실행은 `24.7611%`와
`25.5098%`였으므로, 네 연결 조건을 그대로 다시 돌리기 전에 issue 760에서 사용한
mechanism 계열의 `learned_all` 하나만 분리해 확인합니다.

- CUB main-L0, audited issue-722 teacher, DeiT-Tiny, iBKD λ=0.25
- batch 128, seed 1, learned-all, 300 epoch
- full-data 2-epoch smoke는 H200 issue 767에서 `11/11` gate 통과
- smoke에서는 입력·RNG·student·guidance state hash와 메모리·시간만 확인
- smoke 수치로 checkpoint, epoch, λ 또는 방법을 선택하지 않음
- 별도 300-epoch classification replay에서 validation으로 checkpoint를 선택하고
  official test를 정확히 한 번 평가하며 frozen probe는 수행하지 않음
- 단일 재현이 안정되기 전에는 네 연결 조건 비교를 재개하지 않음

이는 새 과학 결과가 아니라 issue 760의 실행 경로를 진단하는 사후 gate입니다.
단순 smoke 통과만으로 `10.41%`를 이상치로 폐기하지 않습니다.

### Issue 770 재현 뒤 M1-v2

Issue 770 단일 replay는 validation-selected test macro Top-1 `26.3989%`로
issue 760의 `10.4096%`를 재현하지 않았습니다. 결과 archive 감사 전 잠정값이지만,
네 연결 방식 실행 경로를 다시 점검하는 smoke는 먼저 진행할 수 있습니다.

재개 실험은 연결 방식 이외의 차이를 줄이기 위해 기존 동적 종료 M1-v1과 분리한
matched-duration v2입니다. 원래 현상의 canonical source인 issue 727 seed 1에서
관측된 종료 epoch `123`을 결과 확인 전에 공통 horizon으로 고정합니다. 네 방식 모두
epoch `1–123`에는 beta `2.5`, epoch `124–300`에는 beta `0`을 사용합니다. Smoke는
두 full-data epoch만 실행하며 official test와 frozen probe를 열지 않습니다.

- Full v2 계약: `configs/cub200_r50_224_b128_main_l0_ibkd_connection_matched_full_v2.json`
- Smoke v2 계약: `configs/cub200_r50_224_b128_main_l0_ibkd_connection_matched_smoke_v2.json`
- H200 진입점: `scripts/run_main_l0_ibkd_connection_matched_smoke_b128_seed1.sh`

아래 v1 계약은 기존 동적 controller 실행의 역사적 기록이며 v2 본실험에는 사용하지
않습니다.

기존 full v1 계약은
`configs/cub200_r50_224_b128_main_l0_ibkd_connection_full_v1.json`, SHA-256
`1650e76da40c235292fca166b8058c1a9ee279165a275b59122f505f3b7235d8`에
고정했습니다.

## 실행 release와 산출물 규칙

H200 smoke에서 네 분류 경로 `4/4`와 probe 후보 `12/12`의 실행 및 메모리를
확인했습니다. 마지막으로 발견된 문제는 학습이나 평가가 아니라 종료 summary의
teacher audit 카운터 누락이었고, 과학 설정을 바꾸지 않고 회귀 테스트와 함께
수정했습니다. Smoke 수치로 variant, λ, seed, LR, epoch 또는 full 수행 여부를
선택하지 않았습니다.

과학 프로토콜 원본은 위 SHA로 그대로 보존합니다. 실행 release와 H200 분할은
`configs/cub200_r50_224_b128_main_l0_ibkd_connection_full_execution_v1.json`에
별도로 고정했으며 SHA-256은
`145eb88ebba5134f8fcd4b5bf11861465407baff15aaaa41c69809762f0e469c`입니다.
Smoke에서 분류만 seed당 약 `4.46시간`으로 추정되었으므로
10시간 제한의 여유를 확보하기 위해 encoder seed `1/2/3`을 각각 한 issue로
실행합니다.

각 seed의 종료 gate는 다음과 같습니다.

- teacher 다운로드·무결성 감사 `1/1`
- 분류 학습·validation 선택·official test `4/4`
- probe LR 후보 `60/60`, validation 선택 `20/20`, official test `20/20`
- 새 checkpoint `24/24`(분류 4개 + 선택 probe 20개)

Official test는 해당 seed의 분류 4개와 probe 20개의 **모든 validation 선택이 끝난
뒤** 한 번씩만 평가합니다. Seed 1의 결과와 관계없이 seed 2와 3도 동일한 고정
설정으로 수행합니다. Smoke 로그/checkpoint는 제거하고, 최종 Git에는 정리된
본실험 결과와 해시 manifest만 반영합니다.
