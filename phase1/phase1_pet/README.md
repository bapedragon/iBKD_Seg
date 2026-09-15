# Phase 1 — Oxford-IIIT Pet

상태: **37종 분류, batch 64/128 frozen segmentation probe, ALG warm-up 20 진단
완료·감사 통과. 최종 판정 No-Go.**

분류 label로 학습한 DeiT-Tiny encoder를 완전히 고정하고, 공식 trimap으로 모든
방법에 같은 `Conv2d(192, 2, 1)` probe를 학습했습니다. 공식 split은
train/validation/test `2,940 / 740 / 3,669`이며 checkpoint 선택에는 validation만
사용했습니다.

## 본학습 결과

| 범위 | 핵심 결과 | 상세 보고서 |
|---|---|---|
| Batch 64 분류 | LG 38.206%, ALG 32.277%, iBKD-0.25 29.474%, iBKD-0.5 29.528% | [결과](reports/classification/batch64/RESULTS.md) |
| Batch 128 분류 | LG 32.993%, iBKD-0.25 26.716%, iBKD-0.5 24.896%, ALG 22.880% | [결과](reports/classification/batch128/RESULTS.md) |
| Batch 64 frozen probe | LG 83.851%, ALG 82.091%, iBKD-0.25 80.295%, iBKD-0.5 79.733% | [결과](reports/frozen_probe/batch64/RESULTS.md) |
| Batch 128 frozen probe | LG 82.856%, iBKD-0.25 78.256%, iBKD-0.5 77.195%, canonical ALG 63.378% | [결과](reports/frozen_probe/batch128/RESULTS.md) |
| ALG warm-up 20 진단 | ALG probe 80.947%; iBKD 두 설정보다 높음 | [결과](reports/diagnostics/alg_controller_warmup20_b128/RESULTS.md) |

분류 수치는 37-class official-test macro Top-1, probe 수치는 official-test input
resolution 2-class mIoU이며 모두 encoder seed 3개 평균입니다. 정확한 표준편차와
seed별 값은 각 결과 보고서와 CSV에 있습니다.

Canonical batch 128 ALG는 controller가 세 seed 모두 epoch 2에 종료되었습니다.
종료 판정 warm-up만 20으로 바꾼 사후 진단에서 ALG가 회복됐고, batch 64와 함께
보면 iBKD가 LG/ALG보다 공간정보를 더 잘 보존한다는 가설은 지지되지 않았습니다.
최종 해석은 [DECISION.md](DECISION.md)를 기준으로 합니다.

## 본학습 재현 진입점

```bash
# 6방법 × 3 encoder seeds 분류
bash phase1/phase1_pet/scripts/run_full_b64.sh
bash phase1/phase1_pet/scripts/run_full_b128.sh

# 동일 checkpoint의 frozen segmentation probe
bash phase1/phase1_pet/scripts/run_probe_full_b64.sh
bash phase1/phase1_pet/scripts/run_probe_full_b128.sh

# ALG controller warm-up 20 사후 진단
bash phase1/phase1_pet/scripts/run_alg_warmup20_full_b128.sh
```

과학 프로토콜은 [PROTOCOL.md](PROTOCOL.md)와
[`configs/oxford_iiit_pet_phase1_v1.json`](configs/oxford_iiit_pet_phase1_v1.json)에
고정되어 있습니다. ALG 사후 진단의 별도 계약은
[`configs/oxford_iiit_pet_alg_warmup20_full_v1.json`](configs/oxford_iiit_pet_alg_warmup20_full_v1.json)입니다.

## 보존 원칙

- Git에는 정리된 본학습 결과, CSV/JSON 요약, hash·감사 manifest만 둡니다.
- 큰 teacher/student/probe checkpoint와 원시 로그는 각 결과 폴더의
  `checkpoint_release.json` 또는 `artifact_release.json`이 가리키는 검증된 GitHub
  Release에 둡니다.
- Oxford-IIIT Pet 데이터셋, feature cache와 로컬 원시 출력은 Git에 넣지 않습니다.
- 본학습이 완료된 선행 실행 점검용 config·script·결과는 제거했습니다.
