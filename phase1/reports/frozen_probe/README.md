# Phase 1 Pet frozen segmentation probe 결과

이 폴더에는 Oxford-IIIT Pet 공식 trimap을 사용한 frozen spatial probe의 소형
보고서와 정성 panel만 Git으로 추적합니다. 전체 checkpoint, 콘솔 로그와 mask는
GitHub Release에 보존하고 데이터셋과 feature cache는 보존하지 않습니다.

| 분류 batch profile | 상태 | 결과 |
|---|---|---|
| 64 | 6설정 × 3 encoder seed × 5 probe seed 완료 | [batch64/RESULTS.md](batch64/RESULTS.md) |
| 128 | H200 로그상 90/90 완료·산출물 감사 대기 | [batch128/RESULTS.md](batch128/RESULTS.md) |

Batch 64에서는 `iBKD mIoU > matched ALG mIoU`가 지지되지 않았고, batch 128에서는
반대로 iBKD가 ALG보다 `+14.879/+13.817`%p 높았습니다. 그러나 batch 128 ALG는
guidance가 epoch 2에 종료된 checkpoint라 profile 간 방향이 뒤집혔으며, LG가 두
profile 모두 1위입니다. 전체 Phase 1 판단은 batch 128 산출물 독립 감사와 별도 ALG
warm-up 20 사후 진단을 확인한 뒤 기록합니다.

## 결과 반입·감사 절차

공유 ZIP에서 특정 probe 결과만 안전하게 반입합니다.

```bash
PYTHONPATH=src python phase1/scripts/import_probe_archive.py \
  /path/to/result.zip \
  --batch-size 128 \
  --issue-id ISSUE_ID \
  --output-dir phase1/results/raw/oxford_iiit_pet/frozen_probe_v1/batch128 \
  --canonical-bundle-filename \
    phase1_pet_b128_frozen_probe_v1.zip \
  --verify-all-crc
```

그다음 270개 후보 history, validation 선택, 90개 checkpoint, global confusion 기반
metric, test-once, 정성 panel을 독립적으로 다시 검사하고 Git 추적용 결과를 만듭니다.

```bash
PYTHONPATH=src python phase1/scripts/curate_probe_results.py \
  --raw-dir phase1/results/raw/oxford_iiit_pet/frozen_probe_v1/batch128 \
  --report-dir phase1/reports/frozen_probe/batch128
```
