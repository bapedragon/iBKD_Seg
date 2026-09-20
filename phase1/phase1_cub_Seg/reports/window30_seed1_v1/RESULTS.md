# CUB 직접 segmentation window-30 탐색 본학습 결과

상태: **H200 작업 783 완료·로컬 artifact 반입 및 무결성 확인 완료**

이 결과는 `cub200_direct_binary_segmentation_window30_exploratory_full_v1`의 seed 1
탐색 실행입니다. 한 seed 결과이며 설정의 `scientific_result=false`를 유지하므로 논문용
확정 수치나 통계적 결론으로 사용하지 않습니다.

## 실행 계약

- split: `5,394 train / 600 validation / 5,794 official test`
- teacher: scratch ResNet-50, 100 epochs
- student: scratch DeiT-Tiny/16, 100 epochs
- 방법: Vanilla / LG / ALG / iBKD
- checkpoint 선택: validation 2-class mIoU 최고 epoch, 동률이면 앞 epoch
- controller: window 30, threshold `-0.02`, ALG warm-up 0, iBKD warm-up 20
- 주 지표: dataset-level 2-class mIoU

## 결과

| 방법 | 선택 epoch | Val mIoU | Test mIoU | Test bird IoU | Test bird Dice | Test pixel acc. | Guidance 종료 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 68 | 81.991% | 81.653% | 69.721% | 82.160% | 94.410% | 해당 없음 |
| LG | 74 | 81.939% | 81.942% | 70.216% | 82.502% | 94.491% | 종료 없음 |
| **ALG** | **86** | **82.287%** | **82.494%** | **71.094%** | **83.105%** | **94.692%** | 75 |
| iBKD | 91 | 81.906% | 82.298% | 70.772% | 82.885% | 94.628% | 78 |

Teacher는 epoch 56에서 선택됐고 val/test mIoU는 각각 `79.388% / 79.758%`입니다.
Test mIoU의 Vanilla 대비 차이는 LG `+0.289%p`, ALG `+0.842%p`, iBKD
`+0.646%p`입니다. ALG는 iBKD보다 `+0.196%p` 높았습니다.

ALG와 iBKD는 각각 guidance 종료 뒤 11 epoch와 13 epoch 후의 checkpoint가 선택됐습니다.
Controller가 실제로 동작했고, 종료 즉시 성능 선택이 고정된 형태는 아닙니다. 이 실행에서는
`ALG > iBKD > LG > Vanilla` 순서지만 한 seed 탐색 결과라는 한계를 유지합니다.

## 반입한 파일

대용량 원시 결과는 Git에 넣지 않고 다음 로컬 경로에 보관합니다.

`outputs/cub_direct_segmentation_window30_full_v1/`

- `run.log`: 전체 학습 로그
- `source/job_783_result.txt`: H200 작업 783 원본 결과 텍스트
- `setup/install.log`: 실행 환경 설치 로그
- `artifacts/config.json`: 실행 설정
- `artifacts/dataset_audit.json`: 데이터셋 감사 결과
- `artifacts/full_summary.json`: teacher와 네 방법의 전체 최종 결과
- `artifacts/{teacher,vanilla,lg,alg,ibkd}/summary.json`: 방법별 요약
- `artifacts/{teacher,vanilla,lg,alg,ibkd}/history.json`: 100 epoch 이력
- `artifacts/{teacher,vanilla,lg,alg,ibkd}/identity.json`: 실행 식별 정보
- `artifacts/{teacher,vanilla,lg,alg,ibkd}/best.pt`: validation 선택 checkpoint
- `artifacts/{teacher,vanilla,lg,alg,ibkd}/latest.pt`: epoch 100 재개 checkpoint

후속 공간지표 평가는 다음 checkpoint를 사용합니다.

| 대상 | 로컬 파일 | SHA-256 |
|---|---|---|
| Teacher | `artifacts/teacher/best.pt` | `84892fc8b8403512b91ad5a1ed1fb57e6db2416cf76958bbfb9d9b2f3e09bcfb` |
| Vanilla | `artifacts/vanilla/best.pt` | `bb6837b4e414482295565af65bd409c5d2ceef8d3ab01bd424e4659a3059a3f3` |
| LG | `artifacts/lg/best.pt` | `c1110d807332d965b8630c7d3f35739c53ef5dc983e9ea50f40fedb0832bcc19` |
| ALG | `artifacts/alg/best.pt` | `72077b5c12ad30158ff68278ed99ca160fc28b39511affc848241c6b0a6b4a51` |
| iBKD | `artifacts/ibkd/best.pt` | `f72590f157deaf79b9caa74a29e021a6ffd12a2c6fcc43b1a9d84851219fb9a4` |

`full_summary.json`의 파일 SHA-256은
`525fb99ddeff9aa22b5da641e5588bd163412c3c6f95214c4d4cd14465e34286`입니다.
원본 ZIP의 SHA-256은
`9081597420ec852502936dac096a44052e12c5f66518f025c462c3691636041f`였으며,
반입·검증 완료 후 사용자 요청에 따라 삭제했습니다.

## 검증

- ZIP CRC 검사: 통과
- `full_summary.json`: `status=complete`, 네 방법 모두 완료
- 로그 마지막 줄: 전체 최종 결과 JSON 존재
- 오류·NaN·Inf·OOM 흔적: 없음
- 다섯 `best.pt`: `torch.load(weights_only=True)` 통과
- checkpoint 내부 선택 epoch와 `full_summary.json`: 일치
- checkpoint 실제 SHA-256과 요약 파일의 선언값: 일치
- 각 실행의 validation 선택 checkpoint 재로드 검사: 통과
