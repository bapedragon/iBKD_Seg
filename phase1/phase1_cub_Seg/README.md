# Phase 1 CUB 직접 segmentation 탐색

이 폴더는 기존 CUB classification/frozen-probe 실험과 분리해, CUB mask를 직접 정답으로
사용하는 segmentation 확장 가능성을 먼저 확인합니다. Vanilla/LG/ALG/iBKD real-data
smoke와 window-30 seed-1 탐색 본학습까지 완료했습니다. 탐색 실행은
`scientific_result=false`이며 Phase gate 통과나 논문용 확정 결과를 뜻하지 않습니다.

- H200 제출 입력안: [H200_SMOKE_ISSUE.md](H200_SMOKE_ISSUE.md)
- 고정 smoke 설정: [configs/direct_segmentation_smoke_v1.json](configs/direct_segmentation_smoke_v1.json)
- 실행 스크립트: [scripts/run_direct_segmentation_smoke.sh](scripts/run_direct_segmentation_smoke.sh)
- window 30 진단 smoke 설정: [configs/direct_segmentation_window30_smoke_v1.json](configs/direct_segmentation_window30_smoke_v1.json)
- window 30 진단 smoke 실행: [scripts/run_direct_segmentation_window30_smoke.sh](scripts/run_direct_segmentation_window30_smoke.sh)
- 탐색 본학습 프로토콜: [FULL_PROTOCOL.md](FULL_PROTOCOL.md)
- 본학습 설정: [configs/direct_segmentation_full_v1.json](configs/direct_segmentation_full_v1.json)
- 본학습 실행: [scripts/run_direct_segmentation_full.sh](scripts/run_direct_segmentation_full.sh)
- window 30 본학습 설정: [configs/direct_segmentation_window30_full_v1.json](configs/direct_segmentation_window30_full_v1.json)
- window 30 본학습 실행: [scripts/run_direct_segmentation_window30_full.sh](scripts/run_direct_segmentation_window30_full.sh)
- window 30 seed-1 결과: [reports/window30_seed1_v1/RESULTS.md](reports/window30_seed1_v1/RESULTS.md)
- window 30 seed-1 checkpoint release 계약: [reports/window30_seed1_v1/checkpoint_release.json](reports/window30_seed1_v1/checkpoint_release.json)
- seed-1 공간진단 smoke 설정: [configs/spatial_diagnostics_seed1_smoke_v1.json](configs/spatial_diagnostics_seed1_smoke_v1.json)
- seed-1 공간진단 smoke 실행: [scripts/run_spatial_diagnostics_seed1_smoke.sh](scripts/run_spatial_diagnostics_seed1_smoke.sh)
- seed-1 공간진단 본실험 설정: [configs/spatial_diagnostics_seed1_full_v1.json](configs/spatial_diagnostics_seed1_full_v1.json)
- seed-1 공간진단 본실험 실행: [scripts/run_spatial_diagnostics_seed1_full.sh](scripts/run_spatial_diagnostics_seed1_full.sh)

대표 metric은 2-class mIoU이고, foreground IoU·background IoU·foreground Dice·pixel
accuracy를 함께 저장합니다. 3step/validation 8장의 값은 연결 진단에만 사용합니다.

seed-1 공간진단 smoke는 작업 783에서 validation으로 선택된 Vanilla/LG/ALG/iBKD
checkpoint를 고정한 뒤 validation 고정 200장에서 다음 연결만 검사합니다.

- student block11의 선형 `1x1 Conv` part probe
- 같은 block11의 작은 `3x3 Conv + GELU + 1x1 Conv` 비선형 part probe
- teacher layer3와 student block 0~11의 spatial linear CKA
- student attention rollout의 AP, Pointing, foreground mass
- 작업 783에서 이미 감사한 official-test mIoU 참조값

비선형 probe는 선형 probe 용량 부족 여부를 확인하는 보조 진단입니다. 두 probe 모두 smoke에서는
2 epoch만 학습하므로 성능 결론에 사용하지 않습니다. 이번 실행은 official-test 이미지를 다시 열지
않으며, Test mIoU는 기존 `full_summary.json`의 값만 복사합니다. 모든 방법의 전체 수치와 완료
상태는 로그 마지막 줄의 `[CUB_SEG_SPATIAL_SMOKE_FINAL_RESULTS]` JSON에 출력됩니다.

스모크 통과 뒤 고정한 seed-1 공간진단 본실험은 part probe에 전체 train 5,394장과
validation 600장을 사용하고 두 probe를 각각 100 epoch 학습합니다. CKA와 attention도 동일한
validation 600장에서 계산합니다. 스모크에서 Vanilla linear probe의 `lr=0.1` 발산을 확인했기
때문에, 본실험은 batch loss가 `10.0`을 넘거나 non-finite가 된 LR 후보만 실패로 기록하고
선택에서 제외한 뒤 사전에 정한 나머지 LR 후보를 계속 실행합니다. 네 방법 모두 같은 규칙과
LR grid를 사용합니다. 본실험도 encoder seed 1만 보는 탐색 결과이며
`scientific_result=false`입니다.

로컬에서 실제 CUB 데이터 연결을 축소 확인할 때만 아래 명령을 사용할 수 있습니다.
해상도와 표본 수를 명시적으로 줄이므로 H200 smoke 통과를 대신하지 않습니다.

```bash
PYTHONPATH=src .venv/bin/python -m ibkd_seg.cub_segmentation.smoke \
  --device cpu --cpu-small \
  --data-dir data/cub_direct_segmentation \
  --output-dir outputs/cub_direct_segmentation_cpu_small
```
