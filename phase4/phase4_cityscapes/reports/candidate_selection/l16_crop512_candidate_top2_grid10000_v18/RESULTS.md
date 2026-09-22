# Cityscapes L/16 crop512 guided 후보 10,000-step 선별 결과

상태: **4개 H200 pack 완료 · 8/8 안정 · guided 방법별 80,000-step 후보 선택**

2,000-step 선별을 통과한 LG 2개, ALG 2개, iBKD 4개를 seed 1에서 각각
10,000 step 학습했습니다. Cityscapes fine train 2,975장으로 학습하고 val 500장
전체에서 평가했습니다. 모든 실행은 crop512, batch8, 80,000-step LR schedule,
같은 student 초기 state와 같은 입력 순서를 사용했습니다. 순위는 pixel accuracy를
1순위, mIoU를 2순위로 정했습니다.

이 결과는 guided 방법의 하이퍼파라미터 선택용입니다. Vanilla와 표준 logit KD는
포함하지 않았으며, 80,000-step 최종 과학 결과도 아닙니다.

## 완결성 및 감사

- pack 4개에서 후보 8개를 중복 없이 수집
- 8개 모두 10,000/10,000 step과 val 500장 평가 완료
- 8개 모두 `stable`, runtime error와 consistency error 없음
- 모든 실행에서 teacher frozen, student parameter와 optimizer state 유한값 확인
- student 초기 state hash, teacher state hash, 전체 입력 stream hash가 8개에서 동일
- 같은 beta의 LG와 ALG는 최종 metric과 student state hash까지 동일
- 원본 첨부 로그는 앞부분이 잘린 console snapshot이지만 각 pack의 마지막
  machine-readable JSON은 완전하며 파싱과 protocol ID 검사를 통과함

## 전체 결과

| 순위 | 방법 | λ | beta | Pixel accuracy | mIoU | guidance 종료 | 실행시간 | 상태 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **iBKD** | **0.25** | **0.50** | **94.445%** | **69.736%** | step 7,441 | 5.34시간 | 안정·선택 |
| 2 | LG | - | 0.05 | 94.426% | 67.541% | 종료 안 함 | 2.76시간 | 안정·선택 |
| 2 | ALG | - | 0.05 | 94.426% | 67.541% | 종료 안 함 | 2.76시간 | 안정·선택 |
| 4 | iBKD | 0.50 | 0.10 | 94.374% | 65.894% | 종료 안 함 | 5.41시간 | 안정 |
| 5 | LG | - | 0.02 | 94.241% | 66.467% | 종료 안 함 | 2.76시간 | 안정 |
| 5 | ALG | - | 0.02 | 94.241% | 66.467% | 종료 안 함 | 2.76시간 | 안정 |
| 7 | iBKD | 0.25 | 0.25 | 94.060% | 64.155% | step 7,441 | 5.32시간 | 안정 |
| 8 | iBKD | 0.50 | 0.25 | 94.002% | 65.165% | step 7,441 | 5.38시간 | 안정 |

동률 표시는 실제 동률입니다. ALG controller가 10,000 step까지 guidance를 끄지 않은
두 beta에서는 해당 LG와 ALG가 같은 update를 수행해 최종 student state까지
byte-identical했습니다.

iBKD `lambda=0.25, beta=0.5`는 pixel accuracy와 mIoU 모두 전체 1위였습니다.
2,000-step 중간값만 보면 이 조합이 최상위가 아니었으므로, 2,000-step 결과는
발산 제거용 screen으로만 해석하고 최종 후보 선택에는 10,000-step 값을 사용합니다.

## guided 방법별 잠정 선택

| 방법 | 선택 조합 | 선택 근거 |
|---|---|---|
| LG | beta 0.05 | LG 후보 중 pixel accuracy와 mIoU 1위 |
| ALG | beta 0.05 | ALG 후보 중 pixel accuracy와 mIoU 1위 |
| iBKD | lambda 0.25, beta 0.5 | iBKD 및 전체 후보 중 두 metric 모두 1위 |

이 선택은 guided 세 방법 안에서의 80,000-step 진입값입니다. 최종 비교표에는 같은
조건의 CE-only student baseline과 표준 logit KD를 별도 실행해 포함해야 합니다.

## 보존 파일

- `candidate_results.csv`: 여덟 후보의 핵심 수치와 선택 여부
- `candidate_summary.json`: 실행별 metric, class IoU, state hash, controller와 감사 결과
- `source_manifest.json`: pack별 사용자 제공 H200 로그 snapshot의 SHA-256과 terminal JSON hash

저장소 정책에 따라 원시 실행 로그와 체크포인트는 Git에 포함하지 않았습니다.
