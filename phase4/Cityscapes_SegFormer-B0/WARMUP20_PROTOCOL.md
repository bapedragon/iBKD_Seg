# iBKD 두 λ · guidance warm-up 20 재실험

**현재 우선 실행:** 사용자 후속 결정에 따라 [λ=0.25 β4개 smoke](H200_WARMUP20_SMOKE_ISSUE.md)를
먼저 수행하고, 통과 후 같은 네 β를 각각 2k까지 실행해 결과를 확인합니다.
아래 10k는 후속 선별 계획이며 지금 자동 진행하지 않습니다.

2026-09-24 사용자 결정: **iBKD λ=0.25·0.5에만 20 epoch 분량의 종료 보호를 적용하고,
ALG는 warm-up 0을 유지**합니다. 실행 ID는 `cityscapes_b0_ibkd_warmup20_v1`입니다.
[고정 JSON](configs/b0_ibkd_warmup20_v1.json)과 [H200 실행 이슈 2개](H200_WARMUP20_ISSUES.md)를
별도로 마련했습니다. H200 실행은 아직 시작하지 않았습니다.

기존 warm-up 0 iBKD는 step558 또는 744에서 guidance가 종료됐습니다. 이는 현재 종료 규칙에
따른 결과이며, 그 자체로 구현 오류나 증류의 무효성을 뜻하지 않습니다. 이번에는 최소 사용
기간을 늘려 비교합니다. **20이 최적이라는 근거를 확보한 것은 아닙니다.**
기존 결과와 checkpoint는 warm-up 0 pilot으로 보존하고 새 결과와 구분해 보고합니다.

## 종료 기준

1 epoch 분량은 `ceil(2975/16)=186` optimizer step입니다. 확장 목록 loader에서 고유 이미지
2,975장이 정확히 한 번씩 등장하는 데이터 순회를 뜻하지는 않습니다.
매 구간의 **β를 곱하기 전 raw guidance 평균**을 표본 수로 가중해 관측합니다.

| 방법 | 관측 손실 G | 평활한 변화량 s의 종료 조건 | 현재 guidance warm-up |
|---|---|---|---:|
| ALG | locality loss | `s >= -0.02` | 0 |
| iBKD λ=0.25·0.5 | `(1-λ) × alignment + λ × fusion` | `s > -0.02` | 20 |

CE나 val mIoU를 보고 끄지 않습니다. `-0.02`는 이 raw loss 척도의 **절대 변화량 임계값**으로,
손실이 2% 미만 감소한다는 의미가 아닙니다. Window 50도 50 epoch를 기다리는 규칙이 아닙니다.
손실 스케일과 초반 변화에 민감하므로 종료 시점과 성능을 함께 기록합니다.

기존 [controller 구현](../../src/ibkd_seg/phase1/controllers.py)의 수식을 유지합니다.
`G_e`가 e번째 구간 평균일 때 `2 <= e <= 50`에서:

```text
d_e = (G_e - mean(G_1, ..., G_(e-1))) / e
ALG:  s_e = sum(d_2, ..., d_e) / e
iBKD: s_e = sum(d_2, ..., d_e) / (e-1)
```

`e > 50`에서는 `d_e = (G_e - G_(e-50)) / 50`을 사용하고 최근 50개 변화량을 평균합니다.
첫 관측은 변화량이 없으므로 warm-up 0에서도 가장 빠른 종료는 두 번째 관측 끝입니다.

## 20 epoch 보호의 정확한 의미

- 첫 step부터 기존 후보의 β 전체를 사용하며, 손실 관측도 처음부터 누적합니다.
- 1~19번째 관측에서는 종료를 판단하지 않습니다.
- **20번째 관측 끝인 step3720에서 처음 종료 판단이 가능**합니다.
  그 step까지 guidance를 사용하고, 조건을 만족했을 때 **step3721부터 β=0**으로 학습합니다.
- 조건을 만족하지 않으면 21번째 이후 관측에서도 기존 규칙으로 판단합니다. 20에서 강제 종료하지 않습니다.
- 종료 후 teacher·guide 계산과 guide parameter update/weight decay를 생략하고 CE 학습을 계속합니다.
- LR warm-up은 여전히 0입니다. β를 서서히 올리거나 20 epoch 동안 CE만 학습하는 설정이 아닙니다.
- ALG의 warm-up 0, LG·FSKD·Vanilla 설정은 유지합니다.

## 재선별 계획

기존 2k는 약 10.75 epoch 분량이므로 이번 20-epoch 보호 종료 이전입니다.
따라서 새 iBKD에서 2k는 경과 기록으로 두고, **λ별 β 4개 모두 초기화부터 10k까지 실행한 뒤
각각 1개를 선택**합니다. 이는 변경된 schedule에서 비교할 후보를 조기에 제외하지 않기 위한
새 실행 계획이며, 기존 warm-up 0의 2k 상위 2개 선별 결과를 무효화하거나 덮어쓰지 않습니다.

| 조건 | 새 실행의 β 후보 |
|---|---|
| iBKD λ=0.25 | 0.387021, 0.903048, 1.9351, 3.87021 |
| iBKD λ=0.5 | 0.4303, 1.00403, 2.1515, 4.303 |

초기 모델·손실·입력은 바뀌지 않아 기존 25-batch calibration과 β grid를 그대로 사용합니다.
초기화 seed1, adapter seed100001, 학습 RNG seed200001, teacher와 student의 파일 해시,
배치16·crop512, AdamW, **80k LR schedule**, 400 step마다 전체 val500 평가도 유지합니다.
NVIDIA HF ImageNet-only MiT-B0 역변환 출처 예외는 이전 v2와 같습니다.

네 후보가 모두 10k를 완료해야 best val mIoU → 더 이른 best step → candidate ID 순으로
λ별 1개를 고릅니다. Last metric과 2k 경과 지표도 보존합니다. 선택된 warm-up 20 checkpoint에서
동일한 전체 상태를 이어 80k로 진행할 수 있으며, 이번 이슈는 10k에서 종료합니다.

변경 없는 Vanilla·FSKD 각 1개, LG·ALG의 기존 상위 2개씩은 기존 2k 전체 상태에서 이어갑니다.
새 계획의 10k 대상은 **기존 6개 + 새로운 iBKD 8개 = 14개 실행 궤적**입니다.
이 문서의 두 이슈는 새 iBKD 8개만 실행합니다. ALG pack2의 누락된 원본 집계는 별도 확인이
필요하다는 기존 제한을 유지합니다. 최종 비교는 λ 둘 다 포함한 6개 조건으로 유지합니다.
변경 전 pilot과 이번 재탐색의 누적 비용을 함께 보고하며, 방법마다 동일한 탐색 비용을
사용했다고 주장하지 않습니다.

## 코드와 재개의 분리

기존 `b0/`와 공통 controller 소스는 수정하지 않고 별도 `b0_warmup20/` 진입점에서만
iBKD warm-up을 20으로 설정합니다. 기존 source hash를 보존하여 변경 없는 방법의 checkpoint
호환성을 유지합니다. 새 checkpoint에는 별도 프로토콜·코드·warm-up 식별자를 포함합니다.

**기존 warm-up 0 iBKD checkpoint를 새 실행에 넣거나, 꺼진 guidance를 다시 켜서 이어가지 않습니다.**
Warm-up 20끼리만 동일한 student·guide·optimizer·RNG·입력 위치·controller 상태로 재개합니다.
Step3719/3720/3721 손실과 β를 로그에 남기고, 학습 중 첫 20구간의 β 및 조기 종료 여부도 검사합니다.
전체 재개 묶음의 protocol/code signature가 다르면 로딩을 거부합니다.

## 준비 검증과 범위

새 단위 검사는 최소 3720-step 보호, 경계 전후 실제 toy optimizer의 파일 재개·동일성,
20번째 이후에도 guidance를 계속할 수 있음, ALG warm-up 0 유지, 이전 checkpoint 거부,
2k에서 후보를 제외하지 않고 네 후보가 10k를 마친 뒤 1개를 선택함을 확인합니다.
기존 B0 소스 해시가 H200 pack3 기록과 같은지도 검사합니다.

로컬 신규 검사 **7 passed**와 Python/shell 문법·CLI 도움말·문서 링크 검사를 통과했습니다.
가중치 cache가 없을 때도 실패 원인을 마지막 JSON과 보고서에 남기는 경로를 확인했습니다.

현재 로컬에는 실제 upstream·가중치 cache가 없어 새 실제 가중치 CPU 검사는 수행하지 못했습니다.
기존 warm-up 0 H200 검증은 이미 완료했지만 warm-up 20의 GPU 성능·속도 검증은 아닙니다.
H200 실행에는 입력 32-batch 사전 검사와 step2→3 전체 상태 재개·재실행 검사가 포함됩니다.
검사 결과와 성공 범위를 구분하며 GPU 실험 결과는 실행 후 추가합니다.
