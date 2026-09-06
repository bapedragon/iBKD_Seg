# Phase 1 Pet 분류 결과

이 폴더에는 Oxford-IIIT Pet 37개 품종 full-classification 실행에서 나온 소형
결과와 검증 manifest만 Git으로 추적합니다. checkpoint와 전체 콘솔 로그는
`phase1/results/raw/` 아래에 보관하며 `.gitignore`로 제외합니다. 원본 ZIP은 CRC와
SHA-256 검증 및 필요한 결과 반입이 끝난 뒤 삭제할 수 있고, 파일명·byte size·hash와
삭제 여부를 `summary.json`에 기록합니다.

| batch profile | 분류 상태 | frozen probe 상태 | 결과 |
|---|---|---|---|
| 64 | 6설정 × 3 seed 완료 | 본 실험 완료 | [batch64 분류](batch64/RESULTS.md) / [probe](../frozen_probe/batch64/RESULTS.md) |
| 128 | 6설정 × 3 seed 완료 | 대기 | [batch128/RESULTS.md](batch128/RESULTS.md) |

각 batch 보고서에는 H200 원본 JSON/audit 사본, 줄바꿈만 LF로 정규화한 CSV,
seed별 정리표, checkpoint SHA-256과 strict-load 감사 결과가 포함됩니다. 결과를
본 뒤 더 좋은 batch나 iBKD λ만 고르지 않고, 사전 고정한 두 batch profile과 두
λ를 모두 별도로 보고합니다.

두 profile의 같은 seed paired 비교와 controller 종료 epoch 진단은
[BATCH_PROFILE_COMPARISON.md](BATCH_PROFILE_COMPARISON.md)에 있습니다.

## 결과 반입 절차

H200에서 받은 ZIP은 다음 스크립트로 안전한 경로와 CRC를 확인한 뒤, 후속 probe에
필요한 checkpoint와 재현 증거만 raw 영역에 반입합니다.

```bash
PYTHONPATH=src python phase1/scripts/import_classification_archive.py \
  /path/to/result.zip \
  --batch-size 128 \
  --issue-id 702 \
  --output-dir phase1/results/raw/oxford_iiit_pet/full_classification_v1/batch128 \
  --canonical-bundle-filename \
    phase1_pet_b128_classification_issue702_and_b64_frozen_probe_issue706_v1.zip \
  --verify-all-crc
```

그다음 checkpoint를 안전하게 불러와 strict-load·모든 floating tensor 유한값·파일
및 model-state hash를 다시 검사하고 Git 추적용 결과를 생성합니다.

```bash
PYTHONPATH=src python phase1/scripts/curate_classification_results.py \
  --raw-dir phase1/results/raw/oxford_iiit_pet/full_classification_v1/batch128 \
  --report-dir phase1/reports/classification/batch128
```
