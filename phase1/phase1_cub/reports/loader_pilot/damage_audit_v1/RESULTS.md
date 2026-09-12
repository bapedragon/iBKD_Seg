# CUB 이미지 loader crop 손상 감사 v1 결과

상태: **PASS — H200 issue 745, 학습 없음, official test 미접근**

## 한 줄 결론

기존 L0 crop은 평균적으로 새의 visible part와 foreground mask를 약 69%만 남겼지만,
최소 crop 면적을 50%로 제한한 L2는 약 94%를 남겼습니다. 따라서 기존 강한 crop이
공간 단서를 상당히 제거한다는 사실은 확인됐지만, 이것만으로 L2에서 iBKD가 더
좋아진다고 결론 내릴 수는 없습니다.

## 실행 계약

- Derived train 5,394장, 이미지당 5회, profile당 26,970개 crop
- L0/L1은 동일한 crop geometry와 동일 난수를 사용
- Mask, bounding box, part location은 손상 진단에만 사용하고 학습에는 사용하지 않음
- Official test 이미지와 mask는 decode하지 않음
- 실행시간: 91.27초
- 완료 gate: profile `3/3`, finite metric, L0/L1 geometry exact match, test decode `0`

## 결과

| Loader | Visible part 보존 | 모든 visible part 보존 | Foreground mask 보존 | BBox coverage | Near-empty crop | Background-only crop |
|---|---:|---:|---:|---:|---:|---:|
| L0 current strong | 69.3335% | 33.0256% | 69.5441% | 63.5829% | 3.2147% | 2.1691% |
| L1 matched weak | 69.3335% | 33.0256% | 69.5441% | 63.5829% | 3.2147% | 2.1691% |
| L2 conservative spatial | **94.0904%** | **68.7875%** | **93.6316%** | **89.0116%** | **0.0334%** | **0.0037%** |

L2는 L0/L1보다 visible-part 보존이 `+24.7569%p`, foreground-mask 보존이
`+24.0875%p`, bbox coverage가 `+25.4287%p` 높았습니다. 모든 visible part를
한 번에 보존한 crop도 `+35.7619%p` 많았습니다. Near-empty crop은 약 96분의 1,
background-only crop은 약 586분의 1로 감소했습니다.

## 올바른 해석

L0와 L1의 수치가 같은 것은 정상입니다. 두 loader는 crop 범위가 같고 L1에서
color jitter, RandAugment, random erasing만 제거했기 때문입니다. 이번 감사는 crop
기하만 측정하므로 광학 증강의 영향은 후속 학습으로 확인해야 합니다.

L2 결과는 다음 가설을 지지합니다.

> `RandomResizedCrop(scale=[0.08,1.0])`은 CUB에서 새의 일부 또는 전부를 자주
> 제거하며, 위치·형태 정보를 학습하려는 guided 방법에 불리할 수 있다.

아직 확인되지 않은 것은 **상대적인 방법 성능**입니다. L2가 모든 방법을 비슷하게
개선할 수도 있고 LG 우위를 그대로 유지할 수도 있습니다. 따라서 이 감사값으로
loader를 선택하지 않고, 다음 단계에서 L0/L1/L2를 같은 seed와 네 guided 방법으로
학습해 validation Part PCK와 frozen-probe mIoU를 비교합니다.

## 산출물 범위와 제한

사용자가 제공한 console log에서 최종 수치와 완료 gate를 전사했습니다. 전체 H200
결과 archive는 아직 받지 않았으므로 quantile이 포함된 원본 JSON과 정성 crop PNG는
현재 Git 보고서에 포함되지 않았습니다. 출처는 `source_manifest.json`의 byte size와
SHA-256으로 식별합니다.
