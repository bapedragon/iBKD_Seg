# Phase 1 Pet frozen segmentation probe 결과

이 폴더에는 Oxford-IIIT Pet 공식 trimap을 사용한 frozen spatial probe의 소형
보고서와 정성 panel만 Git으로 추적합니다. 전체 checkpoint, 콘솔 로그와 mask는
GitHub Release에 보존하고 데이터셋과 feature cache는 보존하지 않습니다.

| 분류 batch profile | 상태 | 결과 |
|---|---|---|
| 64 | 6설정 × 3 encoder seed × 5 probe seed 완료 | [batch64/RESULTS.md](batch64/RESULTS.md) |
| 128 | 대기 | batch 128 분류 checkpoint로 동일 프로토콜 실행 예정 |

Batch 64의 1차 가설인 `iBKD mIoU > matched ALG mIoU`는 지지되지 않았습니다.
전체 Phase 1 판단은 사전에 고정한 batch 128 probe까지 완료한 뒤 확정합니다.

## 결과 반입·감사 절차

공유 ZIP에서 특정 probe 결과만 안전하게 반입합니다.

```bash
PYTHONPATH=src python phase1/scripts/import_probe_archive.py \
  /path/to/result.zip \
  --batch-size 64 \
  --issue-id 706 \
  --output-dir phase1/results/raw/oxford_iiit_pet/frozen_probe_v1/batch64 \
  --canonical-bundle-filename \
    phase1_pet_b128_classification_issue702_and_b64_frozen_probe_issue706_v1.zip \
  --verify-all-crc
```

그다음 270개 후보 history, validation 선택, 90개 checkpoint, global confusion 기반
metric, test-once, 정성 panel을 독립적으로 다시 검사하고 Git 추적용 결과를 만듭니다.

```bash
PYTHONPATH=src python phase1/scripts/curate_probe_results.py \
  --raw-dir phase1/results/raw/oxford_iiit_pet/frozen_probe_v1/batch64 \
  --report-dir phase1/reports/frozen_probe/batch64
```
