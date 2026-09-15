# CUB 재현성 진단

현재 단계는 `main_l0_v3` iBKD λ=0.25 seed 1의 같은-seed 결과 변동을 먼저
해결하기 위한 사후 진단입니다. 레이어 연결 ablation의 후속 결론은 이 gate가
안정될 때까지 보류합니다.

H200 smoke 진입점:

```bash
bash phase1/phase1_cub/reproducibility/scripts/run_main_l0_ibkd_deterministic_aa_smoke_b128_seed1.sh
```

Smoke는 동일한 2-epoch full-data 실행을 독립 process 두 번 수행하고 자동으로
비교합니다. 성능값은 과학 결과가 아니며 Git에는 smoke 결과나 checkpoint를
보존하지 않습니다. PASS 후 별도의 300-epoch A/A 본실험 계약을 고정합니다.
