# Phase 1 결정 — Oxford-IIIT Pet frozen spatial probe

## 결정

**No-Go: 현재 iBKD 설정이 LG/ALG보다 공간정보를 더 잘 보존한다는 Phase 1 핵심
가설은 Oxford-IIIT Pet에서 지지되지 않았습니다.**

이 결정은 “spatial guidance가 의미 없다”거나 “segmentation 확장을 영구히 포기한다”는
뜻은 아닙니다. 현재 결과만으로 iBKD 고유 구조의 우월성을 전제로 Phase 2로 넘어갈
근거가 없다는 뜻입니다.

## 핵심 근거

| 조건 | 1위 | ALG | iBKD λ=0.25 | iBKD λ=0.5 | iBKD 대 ALG |
|---|---|---:|---:|---:|---|
| Batch 64 canonical | LG 83.851 | 82.091 | 80.295 | 79.733 | 두 λ 모두 낮음 |
| Batch 128 canonical | LG 82.856 | 63.378 | 78.256 | 77.195 | 두 λ 모두 높음 |
| Batch 128 ALG warm-up 20 진단 | LG 82.856 | 80.947 | 78.256 | 77.195 | 두 λ 모두 낮음 |

단위는 test input-resolution 2-class mIoU 백분율입니다.

1. Batch 64에서는 iBKD 두 λ가 matched ALG보다 `-1.796/-2.358`%p 낮았습니다.
2. Batch 128 canonical에서는 iBKD가 ALG보다 높았지만, ALG가 세 seed 모두 epoch
   2에 guidance를 종료한 상태였습니다.
3. ALG의 controller 종료 판정 warm-up만 `0 → 20`으로 바꾼 사후 진단에서 종료
   epoch가 `103/118/137`로 정상화됐고 ALG mIoU가 `63.378 → 80.947%`로
   회복됐습니다. 회복된 ALG는 iBKD 두 λ보다 `+2.691/+3.752`%p 높았습니다.
4. LG는 batch 64와 128 모두 가장 높았습니다. 따라서 iBKD의 LG/ALG 전반 우위는
   어느 profile에서도 성립하지 않습니다.
5. iBKD는 두 batch에서 KD와 Vanilla보다 높았습니다. 이는 grid-aware spatial
   guidance의 가치에 대한 보조 관측이지만 iBKD 고유 구조의 우월성을 입증하지는
   않습니다.

## 해석 제한

- ALG warm-up 20은 결과 확인 뒤 수행한 사후 원인 진단이며 canonical 결과를
  대체하지 않습니다.
- Encoder seed가 3개이므로 formal 통계적 유의성을 주장하지 않습니다.
- 분류 정확도와 probe mIoU가 함께 상승한 ALG 진단만으로 분류 성능과 독립적인
  공간정보 효과를 분리할 수 없습니다.
- Batch 또는 λ 중 결과가 좋은 것만 사후 선택해 주 결과로 보고하지 않습니다.

## 다음 gate

원래 Phase 2는 Phase 1에서 iBKD 공간정보 우위가 관측됐다는 전제의 원인 검증
단계였습니다. 그 전제가 충족되지 않았으므로 **현재 형태의 Phase 2 진입은 보류**합니다.

계속 진행하려면 다음 중 하나를 새 protocol로 사전 고정해야 합니다.

1. CUB-200-2011처럼 분류 라벨과 pixel mask가 함께 있는 외부 데이터셋에서 Phase 1을
   독립 반복해 Pet 특이성 여부를 확인합니다.
2. iBKD를 dense task에 맞게 수정한 새 방법을 설계하고, 기존 iBKD와 구분된 탐색
   실험으로 시작합니다.

결과 근거는 [batch 64 probe](reports/frozen_probe/batch64/RESULTS.md),
[batch 128 probe](reports/frozen_probe/batch128/RESULTS.md),
[ALG warm-up 20 진단](reports/diagnostics/alg_controller_warmup20_b128/RESULTS.md)에
있습니다.
