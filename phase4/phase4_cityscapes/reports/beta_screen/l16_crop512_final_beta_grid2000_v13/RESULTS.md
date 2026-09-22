# Cityscapes L/16 crop512 2,000-step beta 선별 결과

상태: **H200 issue 804 완료 · 원시 결과 감사 통과 · 10,000-step 후보 선별 완료**

후속 iBKD `lambda=0.5` 2,000-step 선별과 전체 10,000-step 결과는 각각
[`../l16_crop512_ibkd_lambda0p5_beta_grid2000_v17/RESULTS.md`](../l16_crop512_ibkd_lambda0p5_beta_grid2000_v17/RESULTS.md),
[`../../candidate_selection/l16_crop512_candidate_top2_grid10000_v18/RESULTS.md`](../../candidate_selection/l16_crop512_candidate_top2_grid10000_v18/RESULTS.md)에 정리했습니다.

이 실행은 Cityscapes train 2,975장과 val 500장을 사용해 LG, ALG, iBKD의 beta
후보를 seed 1에서 2,000 step까지 검사했습니다. 선택 기준은 val pixel accuracy를
1순위, mIoU를 2순위로 고정했습니다. iBKD `beta=0.5`는 바로 앞선 v12 결정성
재현성 검사에서 얻은 동일 조건의 A 실행을 재사용했습니다.

이 결과는 10,000-step 후보를 줄이기 위한 screen입니다. 80,000-step 최종 결과나
방법 간 우열을 주장하는 과학 결과로 사용하지 않습니다.

## 완결성 및 감사

- issue 804 실행: 11개 중 안정 완료 10개, 불안정 조기 중단 1개
- JSON: 19개 모두 파싱 성공
- 종합 consistency error: 0개
- 모든 안정 실행: 2,000/2,000 step, val 500장 평가 완료
- 불안정 실행: iBKD `beta=0.1`, step 108에서 guidance 비율 급증으로 중단
- 체크포인트: 이 선별 실행에는 포함되지 않음
- test split: 사용하지 않음
- 남은 알려진 비결정 연산 경고: CE scalar reduction
  (`nll_loss2d_forward_out_cuda_template`), 이전 재현성 검사에서 update path에는
  영향이 없음을 확인함

## 후보별 결과

| 방법 | beta | 완료 step | Pixel accuracy | mIoU | guidance stop epoch | 판정 |
|---|---:|---:|---:|---:|---:|---|
| LG | 0.02 | 2,000 | 91.836% | 53.608% | 없음 | 안정 |
| LG | **0.05** | 2,000 | **92.331%** | **58.409%** | 없음 | 안정·1위 |
| LG | 0.10 | 2,000 | 78.586% | 20.253% | 없음 | 안정 |
| LG | 0.20 | 2,000 | 88.760% | 42.836% | 없음 | 안정 |
| ALG | 0.02 | 2,000 | 91.836% | 53.608% | 없음 | 안정 |
| ALG | **0.05** | 2,000 | **92.331%** | **58.409%** | 없음 | 안정·1위 |
| ALG | 0.10 | 2,000 | 76.849% | 19.876% | 3 | 안정 |
| ALG | 0.20 | 2,000 | 88.760% | 42.836% | 없음 | 안정 |
| iBKD | 0.10 | 108 | 없음 | 없음 | 없음 | **불안정 중단** |
| iBKD | **0.25** | 2,000 | **91.404%** | **53.582%** | 없음 | 안정·1위 |
| iBKD | **0.50** | 2,000 | **90.956%** | **43.988%** | 없음 | 안정·v12 재사용 |
| iBKD | 1.00 | 2,000 | 77.678% | 21.404% | 없음 | 안정 |

LG와 ALG의 `beta=0.02`, `0.05`, `0.2`는 2,000 step 안에 ALG controller가
guidance를 끄지 않아 같은 학습 경로와 평가값을 냈습니다. ALG `beta=0.1`은 epoch
3에서 guidance를 껐지만, 이미 앞 구간에서 크게 흔들린 뒤라 최종 성능이 낮았습니다.
따라서 ALG의 실제 controller 효과는 10,000-step 실행에서 다시 확인해야 합니다.

## 10,000-step 진출 후보

| 방법 | 1순위 | 2순위 |
|---|---:|---:|
| LG | 0.05 | 0.02 |
| ALG | 0.05 | 0.02 |
| iBKD | 0.25 | 0.50 |

각 방법에서 안정 후보를 pixel accuracy, mIoU 순으로 정렬해 상위 2개를 선택했습니다.
iBKD `beta=0.1`은 불안정하므로 제외하고, `beta=1.0`은 안정했지만 성능이 낮아
제외합니다. 10,000-step 결과에서 방법별 최종 beta를 하나씩 고른 뒤 80,000-step
본학습으로 진행합니다.

## 보존 파일

- `beta_screen_results.csv`: 12개 후보의 핵심 수치와 10,000-step 선택 여부
- `beta_screen_summary.json`: 감사 결과, 통합 순위, 핵심 실행 상태
- `source_manifest.json`: 원본 ZIP과 로컬 원시 파일의 SHA-256·크기
- 원시 결과: `phase4/phase4_cityscapes/results/raw/cityscapes/`
  `l16_crop512_final_beta_grid2000_v13_issue804/`

원시 결과에는 각 실행의 전체 `steps.jsonl`, 클래스별 IoU와 confusion matrix가 든
`summary.json`, 실행 로그, 설정, 데이터 준비·업로드 검사, provenance를 보존했습니다.
원본 ZIP 안의 최상위 결과 텍스트와 `run.log`는 바이트 단위로 같아서 한 사본만
`h200_issue_804.log`라는 이름으로 보존했습니다.
