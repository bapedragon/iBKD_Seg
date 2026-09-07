# Phase 1 Pet batch 128 ALG controller warm-up 20 사후 진단

상태: **분류 3 seed와 frozen probe 15개 완료·독립 감사 통과 — canonical ALG 조기 종료 원인 확인**

이 실험은 H200 작업 `712`에서 batch 128 canonical ALG의 controller 종료 판정
warm-up만 `0`에서 `20` epoch로 변경한 사후 진단입니다. Optimizer LR warm-up,
teacher, student 초기 state, split, batch, loss, `beta=2.5`, threshold `-0.02`,
smoothing window `50`, checkpoint 선택과 probe protocol은 그대로 유지했습니다.
결과를 본 뒤 만든 진단이므로 사전에 LOCK한 canonical Phase 1 결과를 대체하지
않습니다.

## 실행 및 독립 감사

- H200 작업: `712`
- 실행 코드 commit: `4d9ae4411ca9d3a58704919a6b62b3db170975f5`
- split: train 2,940 / validation 740 / official test 3,669
- 분류: ALG × encoder seed `[1,2,3]` × 300 epoch
- probe: encoder 3개 × probe seed 5개 × LR 3개 × 100 epoch
- validation으로 선택한 probe: 15개, LR 후보: 45개
- 공식 test: 분류 checkpoint당 1회, 모든 probe 선택 후 probe당 1회
- 전체 시간: 4,853.515초 = 1시간 20분 53.515초
- 분류 3개 실행 시간: 4,183.059초 = 1시간 9분 43.059초
- probe peak CUDA allocated memory: 134,273,536 byte

공유 ZIP 73,462,153 byte의 전체 member CRC와 SHA-256
`866afc9289144245738f2625c18f5c2864f84180c69db3bb68b462e833bc72a2`를 확인했습니다.
독립 curation은 다음 항목을 다시 검사했습니다.

- 세 분류 history의 300 epoch와 validation best checkpoint 재선택
- 동일 reference teacher와 seed별 canonical 초기 student state 재사용
- teacher 1개, student 3개, probe 15개의 hash·`weights_only=True`·strict-load
- controller warm-up 20, guidance의 첫 20 epoch 활성화와 종료 epoch
- 45개 probe 후보 history 및 validation LR/epoch 선택 재계산
- 선택 전 기록에 probe test 값이 없고 official test가 정확히 15회인지 확인
- 모든 metric의 global confusion 재계산과 계층적 집계 재계산
- 결과 전에 고정한 8개 panel과 8개 원시 mask의 크기·mode·decode 확인

## 분류 결과

| Encoder seed | 선택 epoch | Controller 종료 epoch | Val macro Top-1 | Test macro Top-1 |
|---:|---:|---:|---:|---:|
| 1 | 239 | 103 | 41.216 | 32.059 |
| 2 | 293 | 118 | 40.676 | 31.082 |
| 3 | 273 | 137 | 40.541 | 30.099 |
| **평균 ± 표본 SD** | — | — | **40.811 ± 0.358** | **31.080 ± 0.980** |

Canonical batch 128 ALG는 세 seed 모두 epoch 2에 guidance를 종료하고 test macro
Top-1이 `22.880 ± 0.258%`였습니다. Warm-up 20에서는 종료가 `103/118/137`로
늦어지고 정확도가 `+8.200`%p 회복됐습니다.

| 비교 대상 | Canonical Test macro Top-1 | Warm-up 20 ALG | 차이 |
|---|---:|---:|---:|
| KD | 30.078 | 31.080 | +1.002 |
| LG | **32.993** | 31.080 | -1.913 |
| ALG | 22.880 | 31.080 | **+8.200** |
| iBKD λ=0.25 | 26.716 | 31.080 | +4.364 |
| iBKD λ=0.5 | 24.896 | 31.080 | +6.184 |

분류 seed 원값은 [classification_per_seed.csv](classification_per_seed.csv)에 있습니다.

## Frozen segmentation probe 결과

주 metric은 224×224 동물/배경 2-class global mIoU입니다. Probe seed 5개를 각
encoder 안에서 먼저 평균하고, encoder seed 3개 평균과 표본 표준편차를 백분율로
표시했습니다.

| 설정 | Input mIoU | Foreground IoU | Background IoU | Foreground Dice | Grid mIoU |
|---|---:|---:|---:|---:|---:|
| **ALG warm-up 20** | **80.947 ± 0.258** | **75.462 ± 0.303** | **86.432 ± 0.225** | **86.015 ± 0.197** | **75.448 ± 0.181** |

Encoder seed별 mIoU는 `81.203 / 80.951 / 80.688%`입니다. 세 값 모두 canonical
ALG의 동일 seed보다 높고, iBKD λ=0.25와 λ=0.5의 동일 seed보다도 높습니다.

| 비교 대상 | Canonical Input mIoU | Warm-up 20 ALG | 차이 |
|---|---:|---:|---:|
| KD | 73.765 | 80.947 | +7.182 |
| LG | **82.856** | 80.947 | -1.909 |
| ALG | 63.378 | 80.947 | **+17.569** |
| iBKD λ=0.25 | 78.256 | 80.947 | +2.691 |
| iBKD λ=0.5 | 77.195 | 80.947 | +3.752 |

Batch 64 canonical ALG의 mIoU `82.091%`와 비교하면 차이는 `-1.144`%p로, 기존
batch 128 canonical ALG의 `-18.713`%p 급락 대부분이 사라졌습니다. Probe 원값은
[probe_raw_results.csv](probe_raw_results.csv), encoder별 집계는
[probe_per_encoder_seed.csv](probe_per_encoder_seed.csv)에 있습니다.

## 결론과 해석 범위

1. **Batch 128 ALG 급락의 직접적인 원인은 controller의 epoch-2 종료였다는 근거가
   강해졌습니다.** 오직 종료 판정 warm-up만 바꾸자 분류와 probe가 함께 크게
   회복됐습니다.
2. **회복된 ALG는 iBKD 두 λ보다 probe mIoU가 높습니다.** 따라서 canonical
   batch 128에서 관측한 큰 iBKD–ALG 우위를 iBKD 구조의 공간정보 우월성으로
   해석할 수 없습니다.
3. **LG는 여전히 분류와 probe 모두 가장 높습니다.** Phase 1의 “iBKD가 LG/ALG보다
   공간정보를 더 잘 보존한다”는 핵심 가설은 Pet에서 지지되지 않습니다.
4. Warm-up 20 ALG는 분류 정확도 자체도 iBKD보다 높습니다. 따라서 이 비교만으로
   분류 성능과 분리된 순수 공간정보 보존 효과를 주장할 수도 없습니다.
5. 이 실험은 사후 진단입니다. Canonical ALG를 삭제하거나 교체하지 않고 두 결과를
   함께 공개해야 하며, warm-up 20을 정식 설정으로 사용할 경우 새 protocol에서
   전체 방법을 다시 비교해야 합니다.

Seed가 3개뿐이므로 평균·표준편차와 paired 방향은 기술통계이며 formal p-value나
통계적 유의성을 주장하지 않습니다.

## 정성 결과와 보존

- 고정 정성 panel: [QUALITATIVE.md](QUALITATIVE.md)
- GitHub Release: [artifact_release.json](artifact_release.json)에 teacher/student/probe
  checkpoint와 전체 원시 근거를 보존
- checkpoint audit: [checkpoint_manifest.json](checkpoint_manifest.json)
- 전체 curation 결과: [summary.json](summary.json)
- Git 제외 raw:
  `Phase1_PET/results/raw/oxford_iiit_pet/diagnostics/alg_controller_warmup20_b128_v1/`

사용자 원본 `bapedragon_712.zip`은 수정·삭제하지 않았고, 데이터셋과 feature cache는
Release 및 Git에서 제외했습니다.
