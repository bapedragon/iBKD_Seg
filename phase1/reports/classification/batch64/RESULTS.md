# Phase 1 Pet batch 64 full-classification 결과

상태: **37-way 분류 완료·감사 통과 — frozen segmentation probe 완료**

이 문서는 사전에 LOCK한 Phase 1 v1 프로토콜의 batch 64 profile 결과입니다.
분류 결과만 기록하며, 공간정보 보존 여부는 동일 checkpoint에 공통 frozen probe를
붙인 뒤 별도로 판단합니다.

## 실행 및 무결성

- H200 요청: `#700`
- 실행 코드 commit: `99f2beae43bb7aee07c90c1e81755fc85b850fe9`
- 구성: teacher 1회 + 6설정 × encoder seed 3개 = 19/19 완료, 실패 0
- 전체 경과시간: 26,982.786초 = 7시간 29분 42.786초
- 공식 split: train 2,940 / validation 740 / test 3,669
- 공식 test: 각 선택 완료 checkpoint마다 정확히 1회 평가, 학습·선택 사용 없음
- 동일성 계약: 같은 validation split, 같은 teacher, seed별 동일 student 초기 state 통과
- checkpoint 감사: 19개 모두 SHA-256 일치, `weights_only=True` 로드, strict-load,
  model-state hash, floating tensor 유한값 검사 통과
- 원본 ZIP: 1,995,418,088 byte, 전체 member CRC 통과
- 원본 ZIP SHA-256:
  `57ddd4cffb6232ac07d42689ffe408e25fb7691b302274d1ef33bbf4d4b4658f`
- 정규화한 archive 이름:
  `phase1_pet_b64_full_classification_v1_h200_issue700.zip`
- 원본 archive 보관 상태: 필요한 결과를 검증·반입한 뒤 2026-09-06 로컬에서 삭제

Oxford-IIIT Pet 공식 7,349개 표본의 image–label–trimap 1:1 대응, decode,
split disjointness, trimap 값 `{1,2,3}` 검사도 모두 통과했습니다.

## 정량 결과

주 분류 metric은 원 프로토콜대로 37-class macro Top-1입니다. 아래 값은
encoder seed 3개의 `평균 ± 표본 표준편차`이고, epoch는 validation macro Top-1으로
선택된 seed 1/2/3 checkpoint입니다.

| 설정 | 선택 epoch (s1/s2/s3) | Val macro Top-1 | Test macro Top-1 | Test overall Top-1 | Test Top-5 |
|---|---:|---:|---:|---:|---:|
| Vanilla | 235 / 233 / 270 | 31.171 ± 1.226 | 20.486 ± 0.222 | 20.469 ± 0.242 | 51.776 ± 0.805 |
| KD | 257 / 289 / 293 | 37.297 ± 0.619 | 30.592 ± 1.010 | 30.581 ± 1.013 | 64.577 ± 1.471 |
| **LG** | 260 / 231 / 270 | **46.532 ± 1.407** | **38.206 ± 0.742** | **38.212 ± 0.733** | **73.526 ± 0.888** |
| ALG | 253 / 264 / 291 | 43.333 ± 2.683 | 32.277 ± 1.418 | 32.279 ± 1.438 | 67.902 ± 1.657 |
| iBKD λ=0.25 | 289 / 280 / 261 | 38.604 ± 1.149 | 29.474 ± 0.333 | 29.472 ± 0.319 | 65.304 ± 0.968 |
| iBKD λ=0.5 | 271 / 292 / 291 | 38.108 ± 2.466 | 29.528 ± 2.000 | 29.527 ± 1.999 | 65.458 ± 2.189 |

Teacher는 epoch 275에서 선택됐고 test macro Top-1 `46.182%`, overall Top-1
`46.225%`, Top-5 `79.831%`였습니다.

Seed별 정확한 값은 [per_seed.csv](per_seed.csv), H200에서 생성된 원본 표는
[h200_classification_summary.csv](h200_classification_summary.csv)에 있습니다.

## 결과 해석

1. **batch 64 분류에서는 iBKD가 1위가 아닙니다.** Test macro Top-1 순위는
   `LG > ALG > KD > iBKD λ=0.5 ≈ iBKD λ=0.25 > Vanilla`입니다. 일반적인
   분류 성능 우위를 주장한다면 이 profile은 그 주장을 지지하지 않습니다.
2. **distillation 자체의 효과는 분명합니다.** 모든 guidance 방법이 Vanilla보다
   seed 평균 `+8.988`~`+17.720`%p 높았습니다. 따라서 guided model들이 학습에
   실패한 결과는 아닙니다.
3. **LG가 가장 강하고 안정적입니다.** LG는 세 seed 모두 1위이며 iBKD보다 평균
   약 `8.68`~`8.73`%p 높습니다.
4. **가장 가까운 주 비교인 ALG보다 iBKD가 낮습니다.** iBKD λ=0.25는 세 seed
   모두 ALG보다 낮아 평균 `-2.803`%p이고, λ=0.5는 두 seed에서 낮고 seed 3에서만
   `+0.251`%p여서 평균 `-2.749`%p입니다. 현재 3 seed 결과를 “ALG와 통계적으로
   동급”이라고 단정할 근거는 부족합니다.
5. **두 iBKD λ의 평균은 사실상 같습니다.** λ=0.5 − λ=0.25는 평균
   `+0.054`%p에 불과하고 seed별 부호도 바뀝니다. λ=0.25의 표준편차가 더 작지만,
   이 결과를 보고 한 λ만 사후 선택해서는 안 됩니다.
6. **validation과 official test 사이에는 공통적으로 큰 차이가 있습니다.** 평균
   macro 차이는 방법에 따라 `6.706`~`11.057`%p입니다. 모든 방법이 같은 split과
   selection 정책을 사용했고 leakage 감사도 통과했으므로 비교 계약 위반은
   확인되지 않았습니다. 다만 작은 validation 표본의 변동, 300 epoch 중 최고값
   선택 효과, official split 간 난이도 차이가 가능한 원인이므로 batch 128 및
   probe 결과와 함께 계속 기록해야 합니다.

seed가 3개뿐이므로 이 보고서의 평균·표준편차와 paired 차이는 기술통계입니다.
유의성이나 동등성을 확정하는 검정으로 해석하지 않습니다.

## 공간정보 주장과의 관계

이 분류 결과만으로 iBKD의 공간정보 보존 가설이 기각되거나 입증되지는 않습니다.
핵심 검정은 이때 선택된 **동일 checkpoint의 encoder를 freeze**하고 모든 방법에
동일한 segmentation probe를 학습했을 때의 IoU/Dice입니다.

만약 iBKD가 분류에서는 ALG와 비슷하거나 더 낮은데 frozen probe에서는 여러
encoder seed에 걸쳐 일관되게 높다면, probe 우위가 단순히 “분류 모델 전체가 더
좋아서” 생긴 현상이라는 설명은 약해집니다. 그 경우에는 분류 정확도와 분리된
공간 표현의 선형 복원성 또는 분류–공간정보 trade-off 근거가 됩니다. 반대로
probe에서도 LG/ALG보다 낮다면 현재 Phase 1 설정에서는 iBKD의 공간정보 우위
주장을 지지하기 어렵습니다.

실제 batch 64 frozen probe에서는 LG와 ALG가 iBKD보다 높았습니다. 따라서 이
profile은 기대했던 “분류 성능과 분리된 iBKD 공간정보 우위”를 지지하지 않습니다.
정확한 결과는 [frozen probe 보고서](../../frozen_probe/batch64/RESULTS.md)에 있습니다.
Batch 128 분류도 완료됐으며, 사전 고정한 해당 profile probe는 같은 계약으로
계속 수행합니다. 한 batch나 λ만 사후 선택하거나 hyperparameter를 바꾸지 않습니다.

## 파일 구조

- Git 추적: 이 폴더의 결과 문서, 원본 H200 summary/audit 사본,
  [summary.json](summary.json), [checkpoint_manifest.json](checkpoint_manifest.json)
- Git 제외 raw:
  `phase1/results/raw/oxford_iiit_pet/full_classification_v1/batch64/`
- raw 보존 내용: checkpoint 19개, 개별 summary 19개, H200 로그, import manifest
- 반입 시 제외: ZIP 내부 중복 Oxford-IIIT Pet 데이터, `__MACOSX`, 중복 status/split
  파일
- 삭제한 원본 ZIP은 로컬에서 복구할 수 없으며, 다시 필요하면 H200 결과 원본을
  재수령해야 합니다. 원본의 파일명·byte size·SHA-256·CRC 검증 결과는 남아 있습니다.

전체 콘솔 로그와 checkpoint는 용량 때문에 Git history에 직접 넣지 않고,
[검증된 GitHub Release](checkpoint_release.json)와 raw 경로로 보존했습니다.
