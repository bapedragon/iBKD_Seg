# CUB ResNet-50/224 batch128 guided 3-seed 결과

상태: **H200 issue 730 완료 · issue 727 seed 1과 결합 · 독립 감사 통과**

LG, ALG-w20, iBKD λ=0.25/0.5의 encoder seed `[1,2,3]`가 모두 채워졌습니다.
각 frozen probe 셀은 5개 probe seed를 먼저 평균했고, 아래 `±`는 그 셀 평균을
독립 encoder seed 3개에 대해 계산한 sample SD입니다.

| 방법 | 분류 test macro top-1 (3 encoder seeds) | frozen probe mIoU (3 encoder seeds) |
|---|---:|---:|
| LG | 23.919 ± 1.315% | 73.470 ± 0.673% |
| ALG-w20 | 22.217 ± 1.374% | 70.955 ± 1.150% |
| iBKD λ=0.25 | 24.429 ± 2.334% | 70.318 ± 1.491% |
| iBKD λ=0.5 | 17.081 ± 5.037% | 66.042 ± 2.826% |

## 감사 및 범위

- issue 730 실행시간: 7.43시간
- seed 2·3 분류 8/8, probe 후보 120/120, validation 선택 40/40, official test 40/40
- issue 730의 새 checkpoint 48개 모두 SHA-256, `weights_only=True`, strict load, 유한값 검사 통과
- seed 1 checkpoint는 issue 727 Release를 재사용하며 이 Release에 중복하지 않음
- seed 2·3 원시 checkpoint와 로그는 [검증된 GitHub Release](artifact_release.json)에 보존
- 분류 official test 총 12회(4방법×3seed), probe official test 총 60회(4×3×5)

## 해석

- frozen probe 평균은 **LG가 가장 높습니다**.
- iBKD λ=0.25는 LG 대비 `-3.152%p`, ALG-w20 대비 `-0.637%p`입니다.
- 따라서 이 CUB guided 4방법 블록은 “iBKD가 LG/ALG보다 공간정보를 더 보존한다”는
  가설을 지지하지 않습니다. seed 1에서 보였던 iBKD λ=0.25의 ALG 대비 우위도
  seed 2·3을 포함하면 유지되지 않습니다.
- 이는 **guided 4방법의 3-seed 결과**입니다. Vanilla/KD가 아직 없으므로 최종
  6방법×3seed 매트릭스가 완성됐다고 표기하면 안 됩니다.
