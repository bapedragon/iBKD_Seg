# Phase 1 사후 진단 결과

이 폴더에는 사전에 LOCK한 Phase 1 주 결과를 대체하지 않는 원인 분석 실험을
보관합니다. 진단마다 변경한 항목, canonical 결과와의 관계, test 접근 정책과
해석 범위를 명시합니다.

| 진단 | 변경한 항목 | 상태 | 결과 |
|---|---|---|---|
| Batch 128 ALG controller warm-up 20 | controller 종료 판정 warm-up `0 → 20` | 완료·감사 통과 | [결과](alg_controller_warmup20_b128/RESULTS.md) |

## 결과 반입·감사

```bash
PYTHONPATH=src python Phase1_PET/scripts/import_alg_warmup20_archive.py \
  /path/to/result.zip \
  --issue-id 712 \
  --output-dir \
    Phase1_PET/results/raw/oxford_iiit_pet/diagnostics/alg_controller_warmup20_b128_v1 \
  --verify-all-crc

PYTHONPATH=src python Phase1_PET/scripts/curate_alg_warmup20_results.py \
  --raw-dir \
    Phase1_PET/results/raw/oxford_iiit_pet/diagnostics/alg_controller_warmup20_b128_v1 \
  --report-dir \
    Phase1_PET/reports/diagnostics/alg_controller_warmup20_b128
```

원시 checkpoint와 로그는 Git에서 제외하고 Release asset으로 보존합니다.
