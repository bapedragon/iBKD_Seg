# Phase 1 Pet batch 128 full-classification 결과

상태: **37-way 분류 완료·감사 통과 — frozen probe 로그상 완료·산출물 감사 대기**

이 문서는 사전에 LOCK한 Phase 1 v1 프로토콜의 batch 128 profile 결과입니다.
분류 정확도만 기록하며, 공간정보 보존 여부는 이 checkpoint들의 encoder를 고정한
frozen probe에서 판단합니다.

## 실행 및 무결성

- H200 작업: `702`
- 실행 코드 commit: `3a06aff86007736cad6ec5a9d4d16e425146c0a7`
- 구성: teacher 1회 + 6설정 × encoder seed 3개 = 19/19 완료, 실패 0
- 전체 경과시간: 26,608.978초 = 7시간 23분 28.978초
- 공식 split: train 2,940 / validation 740 / test 3,669
- 공식 test: validation checkpoint 선택이 끝난 뒤 checkpoint당 정확히 1회 평가
- 동일성 계약: batch 64와 같은 split, teacher model-state, seed별 student 초기 state
- checkpoint 감사: 19개 모두 파일 hash, `weights_only=True` 로드, strict-load,
  model-state hash와 floating tensor 유한값 검사 통과
- 원본 ZIP: 2,003,206,323 byte, 전체 37,624개 member CRC 통과
- 원본 ZIP SHA-256:
  `5e931de4de281b893ba2263046c2fc9e2860d0c8e57c6c9073f88a5a2464807c`
- 정규화한 bundle 이름:
  `phase1_pet_b128_classification_issue702_and_b64_frozen_probe_issue706_v1.zip`
- 사용자 원본 `bapedragon_702 1.zip`은 감사 중 수정·삭제하지 않았고, 저장소 내부에는
  중복 보관하지 않았습니다.

Oxford-IIIT Pet 7,349개 표본의 image–label–trimap 1:1 대응, decode, 공식 split
disjointness와 trimap 값 `{1,2,3}` 검사도 통과했습니다.

## 정량 결과

주 분류 metric은 37-class macro Top-1입니다. 아래 값은 encoder seed 3개의
`평균 ± 표본 표준편차`이고, epoch는 validation macro Top-1으로 선택했습니다.

| 설정 | 선택 epoch (s1/s2/s3) | Val macro Top-1 | Test macro Top-1 | Test overall Top-1 | Test Top-5 |
|---|---:|---:|---:|---:|---:|
| Vanilla | 208 / 206 / 233 | 32.117 ± 0.920 | 21.118 ± 0.658 | 21.123 ± 0.651 | 51.231 ± 0.856 |
| KD | 222 / 269 / 251 | 38.829 ± 0.900 | 30.078 ± 1.086 | 30.063 ± 1.086 | 64.741 ± 1.138 |
| **LG** | 284 / 271 / 238 | **40.991 ± 0.978** | **32.993 ± 1.686** | **32.997 ± 1.672** | **68.357 ± 1.918** |
| ALG | 298 / 256 / 287 | 33.018 ± 1.299 | 22.880 ± 0.258 | 22.876 ± 0.246 | 54.974 ± 1.560 |
| iBKD λ=0.25 | 269 / 271 / 256 | 34.730 ± 1.693 | 26.716 ± 0.402 | 26.710 ± 0.386 | 62.006 ± 0.948 |
| iBKD λ=0.5 | 257 / 247 / 276 | 33.604 ± 1.784 | 24.896 ± 1.217 | 24.893 ± 1.211 | 59.807 ± 1.513 |

Teacher는 batch 64와 같은 model-state이며 epoch 275에서 선택됐습니다. Test macro
Top-1은 `46.182%`, overall Top-1은 `46.225%`, Top-5는 `79.831%`입니다.

Seed별 원값은 [per_seed.csv](per_seed.csv), H200 원본 집계는
[h200_classification_summary.csv](h200_classification_summary.csv)에 있습니다.

## 결과 해석

1. **batch 128에서도 LG가 1위입니다.** 순위는
   `LG > KD > iBKD-0.25 > iBKD-0.5 > ALG > Vanilla`입니다.
2. **iBKD는 ALG보다 높지만 KD와 LG보다 낮습니다.** iBKD λ=0.25 − ALG는 세
   seed 모두 양수로 평균 `+3.836`%p, λ=0.5 − ALG도 모두 양수로 평균
   `+2.016`%p입니다. 반면 KD 대비 각각 `-3.362`, `-5.182`%p입니다.
3. **이 profile에서는 λ=0.25가 λ=0.5보다 높습니다.** paired 평균 차이는
   `+1.820`%p이고 세 seed의 부호가 같습니다. 그러나 batch 64 결과를 포함해 두
   λ를 사후 선택하지 않고 모두 보고해야 합니다.
4. **ALG의 큰 하락에는 명확한 controller 진단이 있습니다.** LOCK된 adaptive
   controller는 warmup이 0이고, batch 128 ALG에서 세 seed 모두 epoch 2에 guidance를
   종료했습니다. batch 64 ALG의 종료 epoch는 `118/144/138`입니다. 실행은 고정된
   구현을 그대로 따랐으므로 결과를 고치거나 제외할 수 없지만, ALG가 batch 크기에
   매우 민감했다는 뜻입니다. 이후 수정 실험은 v1 결과와 분리해 최소 history 또는
   warmup을 사전 고정한 ablation으로 해야 합니다.
5. **분류 결과만으로 공간정보를 말할 수 없습니다.** batch 128 checkpoint에도
   동일 frozen probe를 적용해야 iBKD–ALG 공간 표현 비교가 완성됩니다.

seed가 3개뿐이므로 평균·표준편차와 paired 차이는 기술통계입니다. 유의성 또는
동등성을 확정하는 검정으로 해석하지 않습니다.

## Batch profile 비교

batch 64와 128은 데이터, teacher, 초기화와 평가 계약이 같지만 결과 차이가 방법마다
다릅니다. 특히 ALG는 `-9.397`%p, LG는 `-5.212`%p인 반면 Vanilla는
`+0.632`%p입니다. 따라서 “batch 128이 전반적으로 나쁘다”보다 **method × batch
interaction**으로 보는 것이 맞습니다. 두 profile 중 좋은 것만 main 결과로 고르는
것은 금지하며, 상세 표는 [batch profile 비교](../BATCH_PROFILE_COMPARISON.md)에
있습니다.

## 보존 구조

- Git 추적: 이 폴더의 보고서, 원본 summary/audit 사본, 정리 JSON/CSV와 hash manifest
- Git 제외 raw:
  `phase1/results/raw/oxford_iiit_pet/full_classification_v1/batch128/`
- checkpoint 보존: [GitHub Release manifest](checkpoint_release.json)에 고정한
  19개 checkpoint 묶음
- Release asset에서 제외: Oxford-IIIT Pet 데이터셋

후속 batch 128 probe는 Release asset의 byte size와 SHA-256을 확인한 checkpoint로
실행됐고, 전달 로그상 선택·test `90/90`과 최종 pass를 완료했습니다. 현재 잠정
수치는 [batch 128 frozen probe 결과](../../frozen_probe/batch128/RESULTS.md)에
있으며, 전체 결과 bundle을 받은 뒤 독립 감사를 완료합니다.
