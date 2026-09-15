# CUB main-L0 원인 분석

완료된 `main_l0_v3`에서 iBKD λ=0.25의 분류 정확도는 guided 방법 중 높았지만,
마지막 block의 frozen segmentation probe와 Part PCK·CKA는 LG보다 낮았습니다.
이 폴더는 그 차이가 iBKD의 **학습 중 12개 student block을 teacher 3개 stage에
연결하는 방식**에서 왔는지 따로 확인합니다.

이 실험은 완료된 v3 주 결과를 수정하거나 대체하지 않는 사후 원인 분석입니다.
혼동 방지용 전체 계보는 [CUB 실험 계보 색인](../EXPERIMENT_INDEX.md)을 먼저
확인합니다.

## 폴더 구성

- [PROTOCOL.md](PROTOCOL.md): 관찰 분석과 인과 ablation의 질문·고정값·해석 규칙
- `configs/`: smoke 전에 잠근 smoke/full machine-readable 계약
- `scripts/summarize_main_l0_aggregation.py`: issue 727/730의 기존 iBKD checkpoint
  6개에서 실제 aggregation 가중치를 감사·요약
- `scripts/run_main_l0_ibkd_connection_smoke_b128_seed1.sh`: 네 연결 방식의
  2-epoch 분류→frozen probe 실행 경로 점검
- `reports/main_l0_aggregation_checkpoint_audit_v1/`: 학습 없는 기존 checkpoint
  관찰 결과

## 현재 상태

1. 기존 `main_l0_v3` checkpoint 6개의 aggregation 가중치 감사 완료
2. 인과 비교의 full 과학 설정을 결과 전에 고정 완료
3. seed-1 2-epoch smoke 실행기 준비 완료
4. full 실행은 smoke 완료 gate와 실제 시간 확인 전까지 차단

다음 H200 실행 명령은 아래 하나입니다.

```bash
bash phase1/phase1_cub/mechanism_analysis/scripts/run_main_l0_ibkd_connection_smoke_b128_seed1.sh
```

Smoke 수치에는 과학적 의미가 없고 네 방식 중 일부를 탈락시키는 데 쓰지 않습니다.
