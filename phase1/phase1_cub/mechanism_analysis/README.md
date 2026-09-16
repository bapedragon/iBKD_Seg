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
   못했고 연결 조건별 controller 종료 epoch도 달라 인과 해석을 보류
6. [제어 A/A 재현성 gate](../reproducibility/README.md)를 먼저 수행
7. 제어 A/A 결과 뒤에는 기존 네 조건을 그대로 재실행하지 않고, issue 760의
   `learned_all` 경로 하나만 먼저 재현하는 진단을 추가

## Issue 760 `learned_all` 단일 경로 재현

먼저 아래 smoke에서 main-L0, iBKD λ=0.25, batch 128, seed 1의 full-data
2 epoch만 실행합니다. 입력·RNG·student·guidance 상태 해시와 메모리, 예상 300-epoch
시간을 기록하며 official test와 frozen probe는 실행하지 않습니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_learned_all_replay_smoke_b128_seed1.sh
```

Smoke 통과는 실행 경로와 입력 신원이 정상이라는 뜻일 뿐, issue 760의 `10.41%`가
재현되거나 반박됐다는 뜻이 아닙니다. 통과 후 별도 본실험에서 `learned_all` 하나만
300 epoch 학습하고 validation checkpoint 선택 뒤 official test를 한 번 평가합니다.
이 단일 재현이 안정되기 전에는 네 연결 조건 ablation을 재개하지 않습니다.

## 본실험 실행 보류

아래 진입점의 seed-1 실행은 완료됐지만 재현성 및 guidance 기간 혼입 때문에 현재
추가 seed 실행에 사용하지 않습니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_connection_full_b128.sh 1
```

제어 full A/A의 같은-seed 변동 범위를 확인했으며, 다음 gate는 위 `learned_all`
단일 경로 재현입니다. 이 재현이 안정된 뒤에는 네 연결 방식 모두 guidance 기간을
동일하게 고정한 새 protocol을 만들고 다시 비교합니다. 기존 v1 결과만으로 레이어
연결의 효과를 결론내리지 않습니다.

Git에는 코드·프로토콜·정리된 본실험 표·checkpoint 해시 manifest만 반영합니다.
본실험 checkpoint와 원시 결과 archive는 GitHub Release로 보존하고, dataset,
feature cache, smoke 산출물은 보존하지 않습니다.
