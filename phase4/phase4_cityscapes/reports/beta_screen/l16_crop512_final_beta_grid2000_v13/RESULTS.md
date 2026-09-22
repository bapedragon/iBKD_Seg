# Cityscapes L/16 crop512 통합 2,000-step beta 선별 결과

상태: **H200 issue 804와 iBKD λ=0.5 후속 실행 완료 · 원시 결과 감사 통과 · 10,000-step 후보 선별 완료**

이 표는 H200 issue 804의 LG·ALG·iBKD `lambda=0.25` 결과와 후속 iBKD
`lambda=0.5` 결과를 통합합니다. 후속 실행의 상세 감사 결과와 전체 10,000-step
결과는 각각
[`../l16_crop512_ibkd_lambda0p5_beta_grid2000_v17/RESULTS.md`](../l16_crop512_ibkd_lambda0p5_beta_grid2000_v17/RESULTS.md),
[`../../candidate_selection/l16_crop512_candidate_top2_grid10000_v18/RESULTS.md`](../../candidate_selection/l16_crop512_candidate_top2_grid10000_v18/RESULTS.md)에 보존했습니다.

두 실행은 Cityscapes train 2,975장과 val 500장을 사용해 LG, ALG, iBKD의 beta
후보를 seed 1에서 2,000 step까지 검사했습니다. 선택 기준은 val pixel accuracy를
1순위, mIoU를 2순위로 고정했습니다. iBKD `lambda=0.25, beta=0.5`는 바로 앞선
v12 결정성 재현성 검사에서 얻은 동일 조건의 A 실행을 재사용했습니다.

이 결과는 10,000-step 후보를 줄이기 위한 screen입니다. 80,000-step 최종 결과나
방법 간 우열을 주장하는 과학 결과로 사용하지 않습니다.

## 완결성 및 감사

- 통합 후보: 16개 중 안정 완료 15개, 불안정 조기 중단 1개
- issue 804 실행: 11개 중 안정 완료 10개, 불안정 조기 중단 1개
- v12 재사용: iBKD `lambda=0.25, beta=0.5` 안정 완료 1개
- 후속 `lambda=0.5` 실행: 4개 모두 안정 완료
- issue 804 JSON: 19개 모두 파싱 성공
- 후속 `lambda=0.5` 종료 JSON: 파싱과 protocol ID 검사 통과
- 두 실행의 consistency error: 0개
- 모든 안정 실행: 2,000/2,000 step, val 500장 평가 완료
- 불안정 실행: iBKD `lambda=0.25, beta=0.1`, step 108에서 guidance 비율 급증으로 중단
- 체크포인트: 두 선별 실행에는 포함되지 않음
- test split: 사용하지 않음
- 남은 알려진 비결정 연산 경고: CE scalar reduction
  (`nll_loss2d_forward_out_cuda_template`), 이전 재현성 검사에서 update path에는
  영향이 없음을 확인함

## 후보별 결과

| 방법 | λ | beta | 완료 step | Pixel accuracy | mIoU | guidance stop epoch | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| LG | - | 0.02 | 2,000 | 91.836% | 53.608% | 없음 | 안정·10k 진출 |
| LG | - | **0.05** | 2,000 | **92.331%** | **58.409%** | 없음 | 안정·10k 진출 |
| LG | - | 0.10 | 2,000 | 78.586% | 20.253% | 없음 | 안정 |
| LG | - | 0.20 | 2,000 | 88.760% | 42.836% | 없음 | 안정 |
| ALG | - | 0.02 | 2,000 | 91.836% | 53.608% | 없음 | 안정·10k 진출 |
| ALG | - | **0.05** | 2,000 | **92.331%** | **58.409%** | 없음 | 안정·10k 진출 |
| ALG | - | 0.10 | 2,000 | 76.849% | 19.876% | 3 | 안정 |
| ALG | - | 0.20 | 2,000 | 88.760% | 42.836% | 없음 | 안정 |
| iBKD | 0.25 | 0.10 | 108 | 없음 | 없음 | 없음 | **불안정 중단** |
| iBKD | 0.25 | **0.25** | 2,000 | **91.404%** | **53.582%** | 없음 | 안정·10k 진출 |
| iBKD | 0.25 | **0.50** | 2,000 | **90.956%** | **43.988%** | 없음 | 안정·v12 재사용·10k 진출 |
| iBKD | 0.25 | 1.00 | 2,000 | 77.678% | 21.404% | 없음 | 안정 |
| iBKD | 0.50 | **0.10** | 2,000 | **92.161%** | **56.085%** | 없음 | 안정·10k 진출 |
| iBKD | 0.50 | **0.25** | 2,000 | **88.079%** | **44.183%** | 없음 | 안정·10k 진출 |
| iBKD | 0.50 | 0.50 | 2,000 | 77.238% | 19.446% | 없음 | 안정 |
| iBKD | 0.50 | 1.00 | 2,000 | 70.963% | 16.459% | 없음 | 안정 |

LG와 ALG의 `beta=0.02`, `0.05`, `0.2`는 2,000 step 안에 ALG controller가
guidance를 끄지 않아 같은 학습 경로와 평가값을 냈습니다. ALG `beta=0.1`은 epoch
3에서 guidance를 껐지만, 이미 앞 구간에서 크게 흔들린 뒤라 최종 성능이 낮았습니다.
따라서 ALG의 실제 controller 효과는 10,000-step 실행에서 다시 확인해야 합니다.
iBKD는 `lambda=0.25`와 `lambda=0.5`를 별도 실행으로 검사했습니다. lambda별 상위
두 beta를 유지해 10,000-step에서 총 네 iBKD 조합을 비교했습니다.

## 10,000-step 진출 후보

| 방법 | λ | 1순위 beta | 2순위 beta |
|---|---:|---:|---:|
| LG | - | 0.05 | 0.02 |
| ALG | - | 0.05 | 0.02 |
| iBKD | 0.25 | 0.25 | 0.50 |
| iBKD | 0.50 | 0.10 | 0.25 |

LG와 ALG에서는 안정 후보를 pixel accuracy, mIoU 순으로 정렬해 상위 2개를
선택했습니다. iBKD에서는 각 lambda 안에서 상위 beta 2개씩을 선택했습니다.
`lambda=0.25, beta=0.1`은 불안정해서 제외했고, 나머지 미선택 조합은 안정했지만
2,000-step 성능이 낮아 제외했습니다. 이 기준으로 10,000-step 진출 후보는 LG 2개,
ALG 2개, iBKD 4개인 총 8개입니다.

## 보존 파일

- `beta_screen_results.csv`: issue 804와 v12 재사용을 합친 12개 후보의 핵심 수치
- `beta_screen_summary.json`: 위 12개 후보의 감사 결과, 순위, 핵심 실행 상태
- [`../l16_crop512_ibkd_lambda0p5_beta_grid2000_v17/beta_screen_results.csv`](../l16_crop512_ibkd_lambda0p5_beta_grid2000_v17/beta_screen_results.csv):
  후속 `lambda=0.5` 네 후보의 핵심 수치와 10,000-step 선택 여부
- [`../l16_crop512_ibkd_lambda0p5_beta_grid2000_v17/beta_screen_summary.json`](../l16_crop512_ibkd_lambda0p5_beta_grid2000_v17/beta_screen_summary.json):
  후속 네 후보의 감사 결과와 핵심 실행 상태
- `source_manifest.json`: 원본 ZIP과 로컬 원시 파일의 SHA-256·크기
- 원시 결과: `phase4/phase4_cityscapes/results/raw/cityscapes/`
  `l16_crop512_final_beta_grid2000_v13_issue804/`

원시 결과에는 각 실행의 전체 `steps.jsonl`, 클래스별 IoU와 confusion matrix가 든
`summary.json`, 실행 로그, 설정, 데이터 준비·업로드 검사, provenance를 보존했습니다.
원본 ZIP 안의 최상위 결과 텍스트와 `run.log`는 바이트 단위로 같아서 한 사본만
`h200_issue_804.log`라는 이름으로 보존했습니다.
