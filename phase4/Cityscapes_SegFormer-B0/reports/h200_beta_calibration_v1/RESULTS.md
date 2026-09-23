# H200 25-batch β calibration 결과

2026-09-23 사용자 제공 로그를 점검했습니다. 실행 commit `df283908cb82fca70317ccf12b1ed5af9b4c2050`.
LG/ALG와 iBKD 두 경로 모두 **25 batch 완료**, 가중치 유지·teacher 고정·BN/RNG 복원 통과입니다.
Optimizer update·backward·validation·test 사용은 모두 0입니다.

| 조건 | CE 중앙값 | raw guidance 중앙값 | β 후보 |
|---|---:|---:|---|
| LG·ALG | 6.34639454 | 0.96411288 | 0.197479, 0.460784, 0.987394, 1.97479 |
| iBKD λ=0.25 | 6.34639454 | 0.49194235 | 0.387021, 0.903048, 1.9351, 3.87021 |
| iBKD λ=0.5 | 6.34639454 | 0.44246311 | 0.4303, 1.00403, 2.1515, 4.303 |

각 batch의 원시 값으로 중앙값·혼합 loss·반올림·실측 비율을 다시 계산해 모두 일치했습니다.
18개 독립 로그 점검 항목의 결과는 [감사 요약](log_audit.json)에 있습니다.
두 경로의 CE 25개가 정확히 일치합니다. 공통 입력은 400 presentation, 고유 이미지 372장입니다.
Code/config/source asset 식별값, sampler 위치와 실제 train 이름도 대조했습니다.
원격 tensor와 checkpoint 파일을 별도로 내려받아 재실행한 감사는 아닙니다.

전체 실행 486.84초(8분 7초), 손실 측정 합계 31.88초입니다.
LG/ALG 25 batch는 11.31초, iBKD는 20.57초였고 준비 데이터 cache는 재사용하지 않았습니다.
이는 backward가 없는 측정 시간이며 학습 속도로 사용하지 않습니다.

로그로 재구성한 `beta_candidates.json`의 SHA-256도 보고값과 일치했습니다:
`70b37112dcbd0874ce12b85debf97a1113c070717c9ec89258e582a207000559`.
원시 결과 대신 [파생 고정 grid](../../configs/b0_beta_grid_frozen_v1.json)를 다음 실행에 사용합니다.
최적 β를 선택한 결과는 아니며, 다음 단계는 후보별 2k 학습입니다.
