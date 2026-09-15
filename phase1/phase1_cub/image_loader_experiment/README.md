# CUB 이미지 loader 실험

상태: **L0/L1/L2 validation-only 본실험 완료, L2 선택, L2 guided seed-1 예비
본실험 완료. 결과 archive·checkpoint 독립 감사 대기.**

이 폴더는 완료된 `main_l0_v3`를 덮어쓰지 않는 별도 사후 실험입니다. 정확한 계보
구분은 [../EXPERIMENT_INDEX.md](../EXPERIMENT_INDEX.md)를 따릅니다.

## 결과

- 학습 전 augmentation 손상 감사:
  [damage audit](reports/damage_audit_v1/RESULTS.md)
- L0/L1/L2 guided 4방법 validation-only 본실험:
  [전체 표](reports/full_v1_log_snapshot/RESULTS.md)
- 선택된 L2의 guided 4방법 × encoder seed 1 분류·frozen probe 예비 본실험:
  [결과](reports/l2_guided_preliminary_full_seed1_log_snapshot_v1/RESULTS.md)

사전 주 지표인 평균 Part PCK는 L0 `24.6660%`, L1 `21.7878%`, L2
`36.0609%`로 L2가 가장 높았습니다. L2 예비 official-test 결과에서는 분류는
LG `30.0931%`, frozen probe는 ALG-w20 `74.2158%`로 각각 가장 높았습니다.
iBKD의 공간정보 우위는 관측되지 않았습니다.

## 본학습 진입점

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_damage_audit.sh
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_full_b128_seed1.sh
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_full_l1_l2_b128_seed1.sh
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_l2_guided_preliminary_full_b128_seed1.sh
```

세부 계약은 [PROTOCOL.md](PROTOCOL.md)에 있습니다. Git에는 정리된 본학습 표와
manifest만 두며, 결과 archive가 도착하면 checkpoint hash와 machine-readable
원본을 감사해 로그 스냅샷 상태를 확정 보고서로 승격합니다.
