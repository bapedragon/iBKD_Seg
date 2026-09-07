# Phase 1 Pet batch 128 frozen segmentation probe 결과

상태: **본 실험 완료·독립 감사 통과 — canonical ALG 대비 iBKD 우위는 관측됐지만 Phase 1 핵심 가설은 지지되지 않음**

분류 label만으로 학습한 batch 128 encoder 18개를 고정하고, Oxford-IIIT Pet의
공식 pixel trimap으로 동일한 `Conv2d(192, 2, 1)` probe만 학습했습니다. 이 결과는
encoder feature에서 동물/배경 공간정보가 얼마나 선형적으로 복원되는지 비교합니다.

## 실행 및 protocol 감사

- H200 작업: `710`
- 실행 코드 commit: `a5c485d160278e8c96ba8038161bd2367e0e0277`
- LOCK protocol SHA-256:
  `38f743958d1211144495dd9b4c7eb6edd4c12ab1bacbb27c75d38528b3e72143`
- 행렬: 6설정 × encoder seed 3개 × probe seed 5개 = 선택 probe 90개
- 후보: probe마다 LR `[0.01, 0.03, 0.1]` × 100 epoch = 총 270개
- split: train 2,940 / validation 740 / official test 3,669
- 선택: validation `14×14` 2-class mIoU만 사용
- 공식 test: 90개 선택이 모두 끝난 뒤 probe당 정확히 1회
- encoder gradient tensor 0개, probe gradient tensor 2개, parameter 386개
- 전체 시간: 3,396.271초 = 56분 36.271초
- peak CUDA allocated memory: 134,273,536 byte

공유 ZIP 전체 73,462,153 byte의 CRC와 SHA-256
`866afc9289144245738f2625c18f5c2864f84180c69db3bb68b462e833bc72a2`를 확인한 뒤
작업 710 산출물만 분리했습니다. 독립 curation에서 다음을 모두 재검사했습니다.

- 270개 후보의 100-epoch history와 best epoch/LR 선택 재계산
- probe seed별 동일 초기 state와 epoch별 동일 batch order
- 선택 전 기록에 test 값이 없고, 90개 official test가 정확히 한 번인지 확인
- 90개 `.pt`의 SHA-256, `weights_only=True`, strict-load, 유한값과 test-before-write 확인
- 저장 confusion으로 90개 grid/input metric 전부 재계산
- encoder seed → probe seed의 계층적 집계 재계산
- 사전에 고정한 16개 panel과 80개 원시 mask의 크기·mode·decode 확인

## 정량 결과

주 metric은 224×224로 복원한 동물/배경 2-class global mIoU입니다. 각 encoder
seed에서 probe seed 5개를 먼저 평균하고, 아래에는 독립 단위인 encoder seed 3개의
`평균 ± 표본 표준편차`를 백분율로 표시했습니다.

| 설정 | Input mIoU | Foreground IoU | Background IoU | Foreground Dice | Grid mIoU |
|---|---:|---:|---:|---:|---:|
| Vanilla | 60.453 ± 0.426 | 49.656 ± 0.231 | 71.251 ± 0.763 | 66.358 ± 0.206 | 56.999 ± 0.384 |
| KD | 73.765 ± 0.954 | 66.391 ± 1.178 | 81.138 ± 0.741 | 79.797 ± 0.848 | 68.508 ± 0.862 |
| **LG** | **82.856 ± 0.291** | **77.932 ± 0.339** | **87.781 ± 0.245** | **87.597 ± 0.214** | **77.612 ± 0.190** |
| ALG | 63.378 ± 0.633 | 53.198 ± 1.068 | 73.557 ± 0.250 | 69.445 ± 0.913 | 59.320 ± 0.547 |
| iBKD λ=0.25 | 78.256 ± 0.770 | 72.008 ± 0.966 | 84.505 ± 0.574 | 83.724 ± 0.655 | 72.709 ± 0.831 |
| iBKD λ=0.5 | 77.195 ± 0.550 | 70.652 ± 0.716 | 83.738 ± 0.385 | 82.801 ± 0.491 | 71.687 ± 0.571 |

순위는 `LG > iBKD-0.25 > iBKD-0.5 > KD > ALG > Vanilla`입니다. Encoder
seed별 주 metric 원값은 [per_encoder_seed.csv](per_encoder_seed.csv), 90개 probe
원값은 [raw_results.csv](raw_results.csv)에 있습니다.

## Paired encoder-seed 비교

| 비교 | Input mIoU 차이 | seed 1 / 2 / 3 차이 | 방향 일관성 |
|---|---:|---:|---|
| iBKD-0.25 − ALG | +14.879 | +14.755 / +13.718 / +16.163 | 모두 양수 |
| iBKD-0.5 − ALG | +13.817 | +13.955 / +13.022 / +14.474 | 모두 양수 |
| iBKD-0.25 − LG | -4.600 | -4.596 / -5.422 / -3.781 | 모두 음수 |
| iBKD-0.5 − LG | -5.661 | -5.396 / -6.118 / -5.470 | 모두 음수 |
| iBKD-0.25 − KD | +4.492 | +5.128 / +4.369 / +3.978 | 모두 양수 |
| iBKD-0.5 − KD | +3.430 | +4.328 / +3.673 / +2.290 | 모두 양수 |

단위는 percentage point입니다. probe 15개를 독립 표본처럼 취급하지 않았으며,
encoder seed가 3개뿐이므로 formal p-value나 통계적 유의성을 주장하지 않습니다.

## Batch 64 및 ALG warm-up 진단과 함께 본 해석

| 설정 | Batch 64 | Batch 128 | 128 − 64 |
|---|---:|---:|---:|
| Vanilla | 60.188 | 60.453 | +0.266 |
| KD | 72.529 | 73.765 | +1.235 |
| LG | 83.851 | 82.856 | -0.994 |
| ALG | 82.091 | 63.378 | **-18.713** |
| iBKD λ=0.25 | 80.295 | 78.256 | -2.039 |
| iBKD λ=0.5 | 79.733 | 77.195 | -2.538 |

1. **LG가 두 canonical batch profile 모두 1위입니다.** iBKD가 LG/ALG보다
   전반적으로 공간정보를 더 잘 보존한다는 넓은 가설은 지지되지 않습니다.
2. **iBKD–ALG 방향은 batch에 따라 뒤집혔습니다.** Batch 64에서는 iBKD가
   ALG보다 낮지만 batch 128에서는 크게 높습니다. 좋은 profile만 선택해 iBKD
   우위라고 결론 내릴 수 없습니다.
3. **batch 128 canonical ALG의 급락은 controller 조기 종료와 연결됩니다.** 세
   seed 모두 epoch 2에 guidance가 꺼졌습니다. 사후 ALG controller warm-up 20
   진단에서는 종료가 epoch `103/118/137`로 늦어지고 mIoU가 `80.947%`로
   회복됐습니다. 이는 canonical ALG보다 `+17.569`%p이고 iBKD-0.25/0.5보다도
   각각 `+2.691/+3.752`%p 높습니다.
4. **iBKD는 KD와 Vanilla보다 두 λ 모두 높습니다.** Spatial guidance가 logit
   KD보다 공간정보를 더 남길 수 있다는 보조 관측이지만, LG와 정상 작동한 ALG보다
   낮으므로 iBKD 고유의 우월성 근거는 아닙니다.
5. Warm-up 20은 결과를 본 뒤 실행한 원인 진단이므로 canonical ALG를 대체하지
   않습니다. 두 결과를 함께 공개하고, 이후 실험에서 controller warm-up을 사용할
   경우 새 사전 고정 protocol로 모든 비교 방법을 다시 실행해야 합니다.

ALG 진단의 전체 분류·probe 결과는
[warm-up 20 사후 진단](../../diagnostics/alg_controller_warmup20_b128/RESULTS.md)에
분리했습니다.

## 정성 결과와 보존

- 고정 정성 panel: [QUALITATIVE.md](QUALITATIVE.md)
- GitHub Release: [artifact_release.json](artifact_release.json)에 고정한 90개 probe,
  전체 원시 결과, 선택 전 기록, 정성 mask와 로그
- checkpoint audit: [checkpoint_manifest.json](checkpoint_manifest.json)
- 전체 curation 결과: [summary.json](summary.json)
- Git 제외 raw:
  `Phase1_PET/results/raw/oxford_iiit_pet/frozen_probe_v1/batch128/`

사용자 원본 `bapedragon_712.zip`은 수정·삭제하지 않았고, 논리적 bundle 이름은
`phase1_pet_b128_frozen_probe_issue710_and_alg_warmup20_issue712_v1.zip`으로
manifest에 기록했습니다. 데이터셋과 feature cache는 보존 대상에서 제외했습니다.
