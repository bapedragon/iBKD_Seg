# CUB 이미지 loader 실험

상태: **L0/L1/L2 validation-only loader pilot 완료, L2 선택. 결과
archive·checkpoint 독립 감사 대기.**

이 폴더는 완료된 `main_l0_v3`를 덮어쓰지 않는 별도 사후 실험입니다. 정확한 계보
구분은 [../EXPERIMENT_INDEX.md](../EXPERIMENT_INDEX.md)를 따릅니다.

## 결과

- 학습 전 augmentation 손상 감사:
  [damage audit](reports/damage_audit_v1/RESULTS.md)
- L0/L1/L2 guided 4방법 validation-only 본실험:
  [전체 표](reports/full_v1_log_snapshot/RESULTS.md)

사전 주 지표인 평균 Part PCK는 L0 `24.6660%`, L1 `21.7878%`, L2
`36.0609%`로 L2가 가장 높았습니다. 이 결과는 loader가 공간 단서 보존에 미치는
영향을 본 validation-only 탐색 결과이며, iBKD의 우위를 입증하는 결과는 아닙니다.

## 본학습 진입점

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_damage_audit.sh
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_full_b128_seed1.sh
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_full_l1_l2_b128_seed1.sh
```

세부 계약은 [PROTOCOL.md](PROTOCOL.md)에 있습니다. Git에는 정리된 본학습 표와
manifest만 두며, 결과 archive가 도착하면 checkpoint hash와 machine-readable
원본을 감사해 로그 스냅샷 상태를 확정 보고서로 승격합니다. 분류 encoder를 다시
학습해 decoder/probe를 붙이는 L2 후속은 유지하지 않으며, 다음 학습은 별도 직접
segmentation 경로에서 시작합니다.
