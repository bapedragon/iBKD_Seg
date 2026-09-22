# Cityscapes L/16 crop512 iBKD λ=0.5 2,000-step beta 선별 결과

상태: **H200 실행 완료 · 4/4 안정 · 10,000-step 후보 2개 선택**

이 실행은 iBKD 내부 혼합값을 `lambda=0.5`로 고정하고 beta `0.1`, `0.25`,
`0.5`, `1.0`을 seed 1에서 각각 2,000 step 학습했습니다. Cityscapes fine
train 2,975장으로 학습하고 val 500장 전체에서 pixel accuracy와 mIoU를 계산했습니다.
선택 기준은 pixel accuracy를 1순위, mIoU를 2순위로 고정했습니다.

이 결과는 10,000-step 후보를 줄이기 위한 screen입니다. 80,000-step 최종 결과나
방법 간 우열을 주장하는 과학 결과로 사용하지 않습니다.

## 완결성 및 감사

- 네 후보 모두 2,000/2,000 step과 val 500장 평가 완료
- 네 후보 모두 `stable`, runtime error와 consistency error 없음
- teacher frozen, student parameter와 optimizer state 유한값 확인
- 동일 student 초기 state, teacher state, 입력 순서를 사용
- 가이던스 controller는 2,000 step 안에 어느 실행에서도 종료되지 않음
- 원본 첨부 로그는 앞부분이 잘린 console snapshot이지만 마지막 machine-readable JSON은
  완전하며 파싱과 protocol ID 검사를 통과함

## 후보별 결과

| 순위 | λ | beta | Pixel accuracy | mIoU | 완료 step | 실행시간 | 판정 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.5 | **0.10** | **92.161%** | **56.085%** | 2,000 | 1.12시간 | 안정·10k 진출 |
| 2 | 0.5 | **0.25** | **88.079%** | **44.183%** | 2,000 | 1.11시간 | 안정·10k 진출 |
| 3 | 0.5 | 0.50 | 77.238% | 19.446% | 2,000 | 1.10시간 | 안정 |
| 4 | 0.5 | 1.00 | 70.963% | 16.459% | 2,000 | 1.10시간 | 안정 |

beta가 커질수록 이 2,000-step 구간의 성능이 일관되게 낮아졌습니다. 따라서
`beta=0.1`, `0.25`만 10,000-step 후보로 넘겼습니다. 짧은 구간의 순위는 장기
학습 순위를 보장하지 않으므로 이 표는 안정성 확인과 후보 축소에만 사용합니다.

## 보존 파일

- `beta_screen_results.csv`: 네 후보의 핵심 수치
- `beta_screen_summary.json`: 실행별 metric, class IoU, state hash와 감사 결과
- `source_manifest.json`: 사용자 제공 H200 로그 snapshot의 SHA-256과 terminal JSON hash

저장소 정책에 따라 원시 실행 로그와 체크포인트는 Git에 포함하지 않았습니다.
