# CUB main-L0 원인 분석

완료된 `main_l0_v3`에서 iBKD λ=0.25의 분류 정확도는 guided 방법 중 높았지만,
마지막 block의 frozen segmentation probe와 Part PCK·CKA는 LG보다 낮았습니다.
이 폴더는 그 차이가 iBKD의 **학습 중 12개 student block을 teacher 3개 stage에
연결하는 방식**에서 왔는지 확인하는 사후 원인 분석을 분리해 둡니다.

이 결과는 완료된 v3 주 결과를 수정하거나 대체하지 않습니다. 전체 계보는
[CUB 실험 계보 색인](../EXPERIMENT_INDEX.md)을 확인합니다.

## 폴더 구성

- [PROTOCOL.md](PROTOCOL.md): 질문, 고정값, 평가 순서와 해석 규칙
- `configs/*_full_v1.json`: smoke 전에 잠근 과학 프로토콜 원본
- `configs/*_full_execution_v1.json`: smoke 경로 확인 뒤 확정한 seed별 실행 분할과
  산출물 계약
- `scripts/run_main_l0_ibkd_connection_full_b128.sh`: encoder seed 하나의
  분류 4조건 → frozen probe → official test 본실험
- `scripts/summarize_main_l0_aggregation.py`: issue 727/730의 기존 iBKD checkpoint
  6개에서 실제 aggregation 가중치를 감사·요약
- `reports/main_l0_aggregation_checkpoint_audit_v1/`: 학습 없는 기존 checkpoint
  관찰 결과

## 현재 상태

1. 기존 `main_l0_v3` checkpoint 6개의 aggregation 가중치 감사 완료
2. 네 연결 조건과 3 encoder seed의 과학 설정을 결과 전에 고정 완료
3. H200 smoke에서 분류 4/4와 probe 후보 12/12의 계산 경로·메모리 확인 완료
4. smoke 수치로 조건을 고르지 않았으며 smoke 로그와 checkpoint는 Git에서 제거
5. Seed-1 본실험에서 `learned_all` 기준 경로가 기존 main-L0 성능을 재현하지
   못했고 연결 조건별 controller 종료 epoch도 달라 해당 v1 인과 해석을 보류
6. [제어 A/A 재현성 gate](../reproducibility/README.md) 완료: test
   `24.7611/25.5098%`, issue 760의 `10.4096%` 저성능은 재현되지 않음
7. Issue 770 단일 replay도 `26.3989%`로 정상 범위 복귀 확인
8. 공통 guidance 기간을 123 epoch로 고정한 네 연결 방식 classification-only
   v3 본실험 진행 중

## Issue 760 `learned_all` 단일 경로 재현

먼저 아래 smoke에서 main-L0, iBKD λ=0.25, batch 128, seed 1의 full-data
2 epoch만 실행합니다. 입력·RNG·student·guidance 상태 해시와 메모리, 예상 300-epoch
시간을 기록하며 official test와 frozen probe는 실행하지 않습니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_learned_all_replay_smoke_b128_seed1.sh
```

H200 issue 767에서 `11/11` gate가 통과했고, 300 epoch 예상 시간은 약
`3시간 12분`, peak CUDA allocated/reserved는 약 `11.25/16.14 GiB`였습니다.
Smoke 통과는 실행 경로와 입력 신원이 정상이라는 뜻일 뿐, issue 760의 `10.41%`가
재현되거나 반박됐다는 뜻은 아닙니다.

단일 본실험은 아래 진입점으로 `learned_all` 하나만 300 epoch 학습하고 validation
checkpoint 선택 뒤 official test를 정확히 한 번 평가합니다. frozen probe는 수행하지
않습니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_learned_all_replay_full_b128_seed1.sh
```

## 네 연결 방식 matched-duration v2 smoke

Issue 770 replay가 정상 범위로 돌아온 뒤, 연결 방식만 비교하기 위해 가이던스 기간을
canonical issue 727의 `123 epoch`로 네 방식 모두 동일하게 고정한 v2를 추가했습니다.
아래 smoke는 각 방식을 full-data 2 epoch씩 실행해 초기 student·Teacher·split·실제
augmentation 입력 stream과 고정 beta schedule을 검사합니다. Official test와 frozen
probe는 실행하지 않으며 smoke 점수는 과학 결과가 아닙니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_connection_matched_smoke_b128_seed1.sh
```

Issue 776 smoke는 네 분류 경로 `4/4`, 개별 gate `11/11`, 교차 gate `6/6`을
통과했습니다. 같은 student 초기 상태와 epoch별 augmentation 입력 stream, Teacher,
split, beta schedule을 확인했고 official test와 frozen probe는 열지 않았습니다.

## Classification-only v3 본실험

Issue 776 smoke 자체가 이미 classification-only(`official_test=0`,
`frozen_probe=0`)였으므로 추가 smoke는 반복하지 않습니다. 본실험도 원인 분석의
핵심 범위에 맞춰 frozen probe를 제외했습니다. Encoder seed 하나씩 독립 issue로
실행하며, 각 shard는 네 연결 방식의 300 epoch 분류와 validation checkpoint 선택,
네 선택이 모두 끝난 뒤 official classification test만 포함합니다. Seed 1의 결과와
관계없이 seed 2와 3도 같은 설정으로 실행합니다. Segmentation mask archive는
다운로드하거나 읽지 않습니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_connection_classification_full_b128.sh 1
```

Probe를 포함했던 matched-duration v2 full 실행기는 실제 실행 전에 제거했습니다.
필요할 경우 classification 결과를 확인한 뒤 별도 버전의 후속 공간정보 분석으로만
다시 설계합니다.

## 기존 동적 종료 본실험 보류

아래 진입점의 seed-1 실행은 완료됐지만 재현성 및 guidance 기간 혼입 때문에 현재
추가 seed 실행에 사용하지 않습니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_connection_full_b128.sh 1
```

기존 동적 controller v1 결과만으로 레이어 연결의 효과를 결론내리지 않습니다.
이후 비교는 위 classification-only v3만 사용합니다.

Git에는 코드·프로토콜·정리된 본실험 표·checkpoint 해시 manifest만 반영합니다.
본실험 checkpoint와 원시 결과 archive는 GitHub Release로 보존하고, dataset,
feature cache, smoke 산출물은 보존하지 않습니다.
