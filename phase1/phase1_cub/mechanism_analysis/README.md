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
5. 10시간 제한을 피하려고 encoder seed 하나당 H200 issue 하나로 본실험 실행

## 본실험 실행

이번 첫 실행은 seed 1입니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_connection_full_b128.sh 1
```

이후 seed 1 결과와 관계없이 같은 명령의 마지막 인자만 `2`, `3`으로 바꾸어 모두
수행합니다. 각 seed는 분류 checkpoint 4개와 validation에서 선택된 probe
checkpoint 20개, 총 24개의 새 checkpoint를 남깁니다. 모든 validation 선택이
끝나기 전에는 official test를 열지 않습니다.

Git에는 코드·프로토콜·정리된 본실험 표·checkpoint 해시 manifest만 반영합니다.
본실험 checkpoint와 원시 결과 archive는 GitHub Release로 보존하고, dataset,
feature cache, smoke 산출물은 보존하지 않습니다.
