# Cityscapes 구성 로컬 검증

## 실제 Cityscapes 데이터 준비

2026-09-14 사용자 제공 ZIP 두 개를 검사하고 `data/cityscapes`에 train/val을 준비했습니다.
이미지·labelIds 정답 2,975/500쌍 전체의 PNG 디코딩, 2048×1024 해상도, 라벨 범위,
유효 정답, split 중복 검사 및 파일별 byte size·SHA-256 기록을 완료했습니다.
ZIP 전체 CRC도 통과했습니다. Test는 추출하거나 평가하지 않았습니다.

원본 파일 해시 및 H200 전달 절차는 [데이터 준비 기록](DATA_PREPARATION.md)에 있습니다.
새 준비 코드의 상위 폴더 처리, 반복 추출, 기존 데이터 충돌 거부, ZIP CRC 오류 탐지,
경로 이탈·중복 목적지 거부를 확인했으며 기존 테스트와 합쳐 **14개 통과**했습니다.
이는 실제 데이터 준비 검증이며 본학습 성능 측정이 아닙니다.

## DeepLabV3 → Segmenter smoke v1

2026-09-14, 기존과 같은 로컬 CPU 환경에서 실제 DeepLabV3-ResNet101 /
Segmenter-S/16 mask decoder 구조를 사용해 네 방법을 검증했습니다.
모든 가중치는 임의 초기화, crop32×64/batch2/FP32의 명시적인 `--cpu-small` 실행입니다.

- Vanilla/LG/ALG/iBKD **4/4 통과**, 각각 3step + 마지막 update 재시작 재현.
- Student encoder/decoder 초기 hash 동일, guided teacher hash 동일 및 freeze 보존.
- Encoder/decoder/guidance gradient, 19클래스 출력, strict load, optimizer/RNG 복원 검증.
- CPU에서 재시작 후 전체 model/guidance state **bitwise 일치**.
- 합성 48×96 sliding-window, ignore 제외 픽셀 수, pixel accuracy와 mIoU 계산 검증.
- ALG와 iBKD controller 종료 분기 및 상태 복원 진단 통과.
- 기존 Cityscapes 전용 테스트 **10개 통과**.
- Source SHA-256: `3b96030c5df9ee05d854a29940ecb32439057f891fcaddcf7dc99c2891275724`.
- 원시 보고서: `outputs/cityscapes_deeplabv3_segmenter_cpu_smoke_v1/smoke_summary.json`.

위 CPU 결과만으로 768×768/BF16의 CUDA 동작, GPU 메모리, 실제 Cityscapes 정확도 또는
pretrained checkpoint 호환성을 확인한 것은 아닙니다.

이후 사용자 제공 H200 실행 로그에서 다음 완료 표식을 확인했습니다.

```text
[CITYSCAPES_ARCH_SMOKE_DONE] status=passed methods=4/4 scientific_result=false
```

네 방법 모두 3step을 수행했습니다. GPU 합성 smoke가 통과했지만 실제 데이터 정확도나
학습된 teacher의 손실 균형을 검증한 결과는 아닙니다. LG/ALG의 출력 loss 약 `2.6234e10`은
임의 teacher를 사용하는 구조 점검에서 관측한 값이며 수렴 근거로 사용하지 않습니다.
원격 `smoke_summary.json` 파일 자체는 아직 수령하지 않았으므로 상세 GPU 메모리 수치를
이 문서에 추정해서 기재하지 않습니다.

H200 요청은 [#448](https://github.com/Aerodrone-H200/gpu-request/issues/448)로 공식 양식에서
제출했습니다. 실행 코드 commit은 `c5877ec0efa11c43e24d0112f46c1d667c2fe366`입니다.
API로 먼저 생성한 #447은 필수 라벨이 붙지 않아 실행되지 않았으며 중복 방지를 위해 닫았습니다.
GitHub CLI의 `--label provisioning`은 이 계정 권한에서 적용되지 않으므로 이후 H200 요청은
저장소의 공식 issue form으로 제출하고 라벨 및 자동 처리 상태를 확인해야 합니다.

아래 v1/v2는 이전 자체 convolution decoder pilot의 검증 이력입니다.

## v2 — Pixel accuracy 우선 변경 검증

2026-09-14 사용자 요청으로 기본 checkpoint 선택과 첫 보고 지표를 pixel accuracy로
변경했습니다. mIoU는 동일 checkpoint에서 보조 지표로 기록합니다.

- Cityscapes 전용 테스트 **10개 통과**. 정확도는 증가하고 mIoU는 감소하는 두 epoch를
  주었을 때 정확도가 높은 epoch를 선택하고, 그 epoch의 두 지표를 함께 저장함을 검증했습니다.
- ResNet-50 teacher와 Vanilla/LG/ALG/iBKD 합성 학습 **5/5 통과**.
- Pixel accuracy와 mIoU의 checkpoint 재평가 일치 및 CSV/JSON 집계 검증 통과.
- Raw report: `outputs/cityscapes_pixel_accuracy_smoke_v2/smoke_report.json`,
  같은 폴더의 `matrix_summary.json`, `results.csv`.
- v2 source SHA-256:
  `eafb22a777ada0be88afafd3a526d4d6da17d7c1da28dfd123ada412f6a2c362`.
- v2 기본 설정 JSON SHA-256:
  `9eb67b8505059c747d1b4329337406e6262e495ef6e437473e1213923369bcf5`.

실제 Cityscapes 평가와 GPU 검증은 수행하지 않았습니다. 아래 기록은 이전 v1 검증 이력입니다.

## v1 — 최초 구성 검증 이력

검증일: 2026-09-14. macOS CPU, Python 3.13.1, PyTorch 2.11.0,
torchvision 0.26.0, timm 1.0.27에서 확인했습니다.

**아래는 합성 데이터로 검증한 코드 동작입니다. 실제 Cityscapes 성능이 아닙니다.**
H200/CUDA, 512×512 batch 4, BF16의 GPU 메모리는 로컬에서 확인하지 않았습니다.

| 확인 항목 | 결과 |
|---|---|
| Cityscapes 전용 테스트 | 9개 통과 |
| 기존 Phase 1 timing/model/controller 회귀 테스트 | 8개 통과 |
| 실제 ResNet-50 teacher + DeiT-Tiny 4방법 합성 학습 | 5/5 완료 |
| 분리된 CLI subprocess matrix 실행 | 5/5 완료, CSV/JSON 집계 생성 |
| 완료된 matrix에 `--resume` 재실행 | 5/5 계약 확인, 추가 학습 없이 완료 |
| CPU 소형 설정 preflight | Teacher + 4방법, 각 2 optimizer step 통과 |
| iBKD checkpoint 독립 reload/evaluation | 선택 시점의 평가 수치 재현 |

Cityscapes 전용 테스트는 다음 오류 가능성을 확인합니다.

1. 공식 labelId → 19개 trainId와 void 매핑, 잘못된 mask 형식 거부.
2. 이미지별 평균 대신 전체 confusion matrix에서 IoU 계산, ignore 픽셀 제외.
3. 불규칙한 원본 크기 및 crop보다 작은 이미지의 sliding-window 경계·padding 처리.
4. Query chunk attention과 기존 full attention의 출력·입력 gradient·파라미터 gradient 일치.
5. 데이터 변경 해시 탐지 및 공식 split 개수와 다른 입력의 audit 거부.
6. 이미지와 마스크 변환의 일치 및 동일 seed/epoch에서 같은 crop 재현.
7. 잘못된 평가 stride와 실제 프로토콜에 합성 데이터 사용 시 거부.
8. Teacher 1개 + 4방법 × 3 seed 구성 및 teacher checkpoint 공유.
9. 연속 2 epoch와 1 epoch 후 재개의 최종 student 가중치 **bitwise 일치**,
   재개 중 학습률 변경 시 거부.

재현 명령은 저장소 루트에서 실행합니다.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_cityscapes.py -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_phase1_timing.py -v
PYTHONPATH=src .venv/bin/python -m ibkd_seg.cityscapes.run smoke \
  --output-dir outputs/cityscapes_new_smoke --device cpu --threads 2
```

원시 합성 출력은 Git에서 제외한 다음 로컬 경로에 있습니다.

- `outputs/cityscapes_synthetic_smoke_v2/smoke_report.json`
- `outputs/cityscapes_preflight_cpu_v1.json`
- `outputs/cityscapes_cli_matrix_validation_v1/matrix_summary.json`
- `outputs/cityscapes_cli_matrix_validation_v1/results.csv`

합성 smoke는 crop 32×64, batch 2, FP32, decoder width 8을 사용했습니다.
실제 모델 backbone과 guidance의 채널·층 구조는 동일하지만, 이 크기의 통과로
기본 GPU 설정의 메모리나 학습 성능을 추정하지 않습니다.

검증 시점 source SHA-256:
`5519324943c047d865154058dcaeb68e5a4c39a54db7e8a024cd438c10239664`

기본 설정의 정규화된 JSON SHA-256:
`bca3c2ff37ee30f9e1d5dd3bdf7270ef74fdb38a7b80c34d5178f6ef41e1d25f`

원시 보고서의 합성 mIoU는 경로 검증용 값이며, 방법 간 성능 비교에 사용하지 않습니다.
