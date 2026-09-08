# CUB ResNet-56/32 v2 guided shard 결과

상태: **H200 issue 716 완료 · 독립 감사 통과 · 최종 v3 비교에서는 제외**

이 실행은 당시 잠근 v2 계약 아래에서는 완전한 scientific guided shard입니다.
다만 이후 최종 CUB 프로토콜을 ResNet-50/224 scratch Teacher v3로 교체했으므로,
이 결과를 v3의 Vanilla/KD/LG 또는 향후 ALG/iBKD 결과와 합치면 안 됩니다.

## 실행 완결성

- Teacher 1개, guided encoder 3방법 × 3 seed = 9개
- Probe LR 후보 135개, validation 선택 45개, official-test 평가 45/45
- Teacher/encoder/probe checkpoint 55개 모두 파일 hash, strict load, 유한값 감사 통과
- train/validation/test 5,394 / 600 / 5,794 및 validation split hash 일치
- 전체 실행시간: 6.86시간
- 전체 checkpoint·로그 보존: [검증된 GitHub Release](artifact_release.json)

## Teacher

| 구조 | 선택 epoch | validation macro top-1 | test macro top-1 |
|---|---:|---:|---:|
| ResNet-56/32 scratch | 275 | 36.000% | 35.432% |

## Guided 분류 결과

| 방법 | test macro top-1 (3 encoder seeds, mean ± sample SD) | ALG-w20 대비 paired 차이 |
|---|---:|---:|
| ALG-w20 | 45.879 ± 0.501% | 기준 |
| iBKD λ=0.25 | 45.848 ± 0.250% | -0.031%p |
| iBKD λ=0.5 | 45.449 ± 1.899% | -0.430%p |

## Frozen probe 결과

독립 반복 단위는 encoder seed입니다. 각 encoder의 5개 probe seed 평균을 먼저 낸
뒤 3개 encoder seed의 mean ± sample SD를 계산했습니다.

| 방법 | test input-224 mIoU | ALG-w20 대비 matched-seed 차이 |
|---|---:|---:|
| ALG-w20 | 77.882 ± 0.101% | 기준 |
| iBKD λ=0.25 | 77.546 ± 0.247% | -0.336%p |
| iBKD λ=0.5 | 77.538 ± 0.347% | -0.344%p |

## 해석

구버전 v2 조건에서는 ALG-w20이 두 iBKD보다 probe mIoU가 약 0.34%p 높았습니다.
분류 정확도는 ALG-w20과 iBKD λ=0.25가 거의 같았지만, 공간 probe에서는 iBKD의
우위가 관찰되지 않았습니다. 다만 차이는 seed별로 완전히 일관되지는 않았고
(seed 2에서는 두 iBKD가 ALG-w20보다 근소하게 높음), 세 방법만 포함한 shard라
Vanilla/KD/LG까지 포함한 전체 순위도 알 수 없습니다.

가장 중요한 제한은 이것이 **ResNet-56/32 v2 결과**라는 점입니다. 현재 최종
프로토콜은 ResNet-50/224 v3이므로 이 수치는 참고·민감도 분석으로만 보존하고,
iBKD 핵심 주장에 대한 최종 판정은 v3의 6방법 × 3 seed 결과로 내려야 합니다.
