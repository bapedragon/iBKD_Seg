# Phase 3 CUB 직접 segmentation 탐색

이 폴더는 기존 CUB classification/frozen-probe 실험과 분리해, CUB mask를 직접 정답으로
사용하는 segmentation 확장 가능성을 먼저 확인합니다. 현재 추가된 것은
Vanilla/LG/ALG/iBKD **real-data smoke**뿐이며 Phase gate 통과나 본실험 시작을 뜻하지 않습니다.

- H200 제출 입력안: [H200_SMOKE_ISSUE.md](H200_SMOKE_ISSUE.md)
- 고정 smoke 설정: [configs/direct_segmentation_smoke_v1.json](configs/direct_segmentation_smoke_v1.json)
- 실행 스크립트: [scripts/run_direct_segmentation_smoke.sh](scripts/run_direct_segmentation_smoke.sh)

대표 metric은 2-class mIoU이고, foreground IoU·background IoU·foreground Dice·pixel
accuracy를 함께 저장합니다. 3step/validation 8장의 값은 연결 진단에만 사용합니다.

로컬에서 실제 CUB 데이터 연결을 축소 확인할 때만 아래 명령을 사용할 수 있습니다.
해상도와 표본 수를 명시적으로 줄이므로 H200 smoke 통과를 대신하지 않습니다.

```bash
PYTHONPATH=src .venv/bin/python -m ibkd_seg.cub_segmentation.smoke \
  --device cpu --cpu-small \
  --data-dir data/cub_direct_segmentation \
  --output-dir outputs/cub_direct_segmentation_cpu_small
```
