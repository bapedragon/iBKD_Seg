# CUB 이미지 loader L0/L1/L2 결과 — 로그 스냅샷

상태: **세 profile 완료 로그 확인, 결과 archive·checkpoint 감사 대기**

이 문서는 사용자가 전달한 H200 로그의 최종 marker를 그대로 표로 옮긴 잠정
스냅샷입니다. L0/L1/L2 모두 encoder seed 1, guided 네 방법, validation-only
조건이며 official test 평가는 0회입니다. 결과 archive가 전달되면 checkpoint와
machine-readable 산출물의 SHA-256을 감사한 뒤 archive 기반 최종 보고서로
교체합니다.

## Loader별 평균

모든 지표는 높을수록 좋습니다. 단, loader 선택에는 사전에 고정한 **Part PCK
평균만 주 지표**, segmentation mIoU 평균만 정확한 동률 시 tie-break로
사용합니다. Classification, CKA와 attention은 선택 지표가 아닙니다.

| Loader | 분류 val macro top-1 (%) | Seg val mIoU (%) | Part PCK@0.1 (%) | CKA block 11 | Attention AP (%) | Pointing (%) | 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 strong | 26.7083 | 71.3057 | 24.6660 | 0.404411 | 24.2827 | **50.5000** | 3:24:14 |
| L1 weak | 19.2500 | 70.3095 | 21.7878 | 0.395629 | **29.5450** | 50.1250 | 3:08:18 |
| **L2 conservative spatial** | **28.1250** | **73.2966** | **36.0609** | **0.483546** | 26.9122 | 46.2500 | 3:13:15 |

사전 선택 규칙상 L2가 선택됩니다. L2의 Part PCK 평균은 L0보다 `+11.3949%p`,
L1보다 `+14.2731%p` 높습니다. Tie-break인 segmentation mIoU도 L0보다
`+1.9909%p`, L1보다 `+2.9871%p` 높습니다.

## 방법별 전체 결과

| Loader | 방법 | 분류 val (%) | Seg mIoU (%) | Part PCK (%) | CKA b11 | Attn AP (%) | Pointing (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| L0 | LG | 27.0000 | 74.3031 | 31.5798 | 0.485855 | 25.8074 | 51.5000 |
| L0 | ALG-w20 | 23.6667 | 69.7727 | 21.2269 | 0.422925 | 22.6967 | 48.5000 |
| L0 | iBKD λ=0.25 | 28.3333 | 71.1646 | 25.2269 | 0.388379 | 25.1118 | 51.5000 |
| L0 | iBKD λ=0.5 | 27.8333 | 69.9825 | 20.6303 | 0.320483 | 23.5147 | 50.5000 |
| L1 | LG | 21.0000 | 72.3174 | 27.0924 | 0.463888 | 21.9471 | 38.5000 |
| L1 | ALG-w20 | 19.8333 | 72.7121 | 27.8571 | 0.464797 | 30.8529 | 52.0000 |
| L1 | iBKD λ=0.25 | 17.3333 | 67.6995 | 14.5210 | 0.329672 | **33.7568** | **61.5000** |
| L1 | iBKD λ=0.5 | 18.8333 | 68.5088 | 17.6807 | 0.324160 | 31.6230 | 48.5000 |
| L2 | **LG** | **31.5000** | **73.9186** | **41.7227** | **0.514921** | **28.9222** | 48.0000 |
| L2 | ALG-w20 | 28.6667 | 73.7160 | 39.0000 | 0.508612 | 27.2832 | 41.5000 |
| L2 | iBKD λ=0.25 | 27.6667 | 73.6832 | 33.3529 | 0.495034 | 28.0149 | **52.0000** |
| L2 | iBKD λ=0.5 | 24.6667 | 71.8685 | 30.1681 | 0.415618 | 23.4283 | 43.5000 |

굵은 값은 각 loader 내부 최고값입니다. Attention 지표는 서로 순서가 엇갈리며
사전 선택 규칙에도 들어가지 않으므로 loader 결정에 사후 사용하지 않습니다.

## 해석

- L2는 네 방법 평균 기준 Part PCK, segmentation mIoU, CKA와 분류 성능을 모두
  높였습니다. 따라서 강한 crop을 완화하면 CUB의 공간 단서가 더 잘 보존된다는
  loader 수준의 근거는 분명합니다.
- 하지만 L2 안에서도 LG가 Part PCK, segmentation mIoU와 CKA에서 가장 높고,
  ALG도 두 iBKD보다 Part PCK가 높습니다. 따라서 이 결과는 **iBKD 고유의 공간
  우위**를 지지하지 않습니다.
- 현재 비교는 encoder seed 1 하나뿐입니다. 다음 `L2 × guided 4방법 × seed 1`
  실행도 6방법×3seed 최종 확증실험이 아니라 선택 loader를 official test까지
  연결하는 단일-seed 예비 결과로 보고해야 합니다.
- 결과 archive가 도착하기 전에는 로그 수치와 completion marker만 확인된 상태입니다.
  Archive 감사 전에는 Stage C 본학습을 시작하지 않고 비과학적 smoke만 수행합니다.

원자료를 Git에 복사하지 않고, 로그 식별 정보는
[source_manifest.json](source_manifest.json)에, 분석용 원수치는
[profile_method_results.csv](profile_method_results.csv)와
[profile_means.csv](profile_means.csv)에 보존합니다. 선택 상태와 핵심 차이는
[summary.json](summary.json)에서도 확인할 수 있습니다.
