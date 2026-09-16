# H200 이슈 입력안 — Cityscapes 압축 해제·실제 데이터 smoke

제목: `[Request]: Cityscapes 압축 해제 및 DeepLabV3 → Segmenter real-data smoke`

[공식 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)에
다음 항목을 입력합니다. 이 문서는 제출용 입력안이며 이슈를 자동 등록하지 않습니다.

### 사용자 ID (Username)

bapedragon

### 실행할 코드의 GitHub 링크

https://github.com/bapedragon/iBKD_Seg.git

### 코드 실행 명령어

```bash
bash phase4/phase4_cityscapes/scripts/run_cityscapes_real_smoke.sh
```

### 사용할 이미지 선택

pytorch/pytorch:latest

### 사용 언어 (Language)

Python

### GPU 할당량 (MIG 갯수)

7

`7`은 물리 H200 GPU 1장입니다. 실제 모델의 crop768/batch2/BF16과 iBKD cross-attention을
확인하므로 앞선 파일 확인용 이슈의 MIG `1`과 구분합니다.

## 한 번의 이슈에서 실행하는 순서

1. `/app/data/chaoyang`에 있는 두 ZIP의 byte size·SHA-256·CRC를 다시 확인합니다.
2. Train·val을 `/app/scratch/cityscapes_real_smoke_v1`에 추출합니다.
   원본 ZIP과 기존 Chaoyang 데이터는 읽기만 합니다. `__MACOSX`와 test는 추출하지 않습니다.
3. 이미지와 labelIds 정답 전체 2,975/500쌍을 디코딩해 해상도·라벨을 확인하고 파일별 해시를 기록합니다.
4. **DeepLabV3-ResNet101 → Segmenter-S/16**으로 Vanilla/LG/ALG/iBKD를 순차 실행합니다.
5. 네 방법은 같은 train 6장·paired crop을 batch2로 나누어 각각 3step 수행합니다.
   마지막 step은 checkpoint를 strict-load하고 optimizer/RNG/controller를 복원한 뒤 한 번 재실행해 비교합니다.
6. 같은 val 2장을 원본 1024×2048에서 window768/stride512로 평가합니다.
   Void를 제외한 pixel accuracy와 mIoU, 손실, gradient, step 시간, GPU 메모리를 기록합니다.

압축 해제와 전체 데이터 검사는 CPU·디스크 작업입니다. 다음 모델 점검에서 GPU를 사용합니다.
오류가 나면 이후 단계를 중단하며, 방법별로 batch나 crop을 자동 축소하지 않습니다.

## 해석 범위

**실제 Cityscapes 이미지·정답을 사용하지만, 가중치는 임의 초기화입니다.**
Teacher/encoder pretrained weight를 다운로드하거나 학습된 teacher를 불러오지 않습니다.
따라서 3step과 val 2장의 점수는 데이터 연결·학습 연산·평가 코드의 진단값이며,
성능 비교, checkpoint 선택, 500장 전체 val 결과, 본학습 수렴 근거로 사용하지 않습니다.
기존 torchvision DeepLabV3와 Segmenter 계열 구조를 유지하고 공통 smoke 엔진을 재사용합니다.

## 고정 조건

- 설정: `configs/deeplabv3_segmenter_real_smoke_v1.json`.
- Crop768×768, batch2, BF16, SGD lr0.01/momentum0.9/wd0, gradient clip1.
- 데이터 ID를 정렬한 첫 train 6장·val 2장 사용. 결과를 보고 표본을 교체하지 않습니다.
- Train resize scale0.5–1.0, paired crop/flip, 최대 category ratio0.75, crop 시도10회.
  이 값들은 이번 연결 점검용이며 본학습 공개 recipe를 확정하지 않습니다.
- 동일 raw RGB에서 teacher는 ImageNet mean/std, student는 mean/std0.5를 적용합니다.
- Student 초기 state·입력 tensor hash, guided teacher 초기 state·freeze 보존을 방법 간 대조합니다.
- LG/ALG/iBKD의 기존 loss·controller 규칙 및 iBKD λ0.25를 유지합니다.
- 3step으로는 guidance의 장기 종료 시점을 관측하지 않습니다.
  고정 손실을 넣는 별도 controller 진단은 실제 학습 결과와 구분해서 저장합니다.
- Test 이미지는 ZIP CRC 점검만 하고 모델 입력·평가에 사용하지 않습니다.

## 결과와 완료 표식

출력 루트는 `/app/output/cityscapes_deeplabv3_segmenter_real_smoke_v1`입니다.

- `run.log`, `upload_check.json`: 전체 실행 로그, 업로드된 ZIP 확인 결과.
- `manifest.json`, `preparation.json`: train/val 파일·해시 및 원본 ZIP·추출 기록.
- `artifacts/smoke_summary.json`: 네 방법의 최종 요약.
- `artifacts/<method>/summary.json`, `resume_checkpoint.pt`: 방법별 진단과 재개 점검 checkpoint.

로그의 최종 통과 표식:

```text
[CITYSCAPES_REAL_SMOKE_DONE] status=passed methods=4/4 scientific_result=false
```

압축을 푼 `/app/scratch` 데이터는 해당 컨테이너의 작업용입니다. 이후 실행도 공유 폴더의
동일 ZIP에서 자동 준비할 수 있습니다. 결과는 `/app/output`에 저장합니다.
이미 결과가 있는 출력 폴더는 덮어쓰지 않으므로 같은 환경에서 재실행할 때는
`CITYSCAPES_REAL_SMOKE_OUTPUT_DIR`로 새 폴더를 지정합니다.

## 로컬 재현 명령

2026-09-16 실제 데이터 CPU smoke 4/4, 기존 합성 smoke 4/4, 관련 테스트 16개를 통과했습니다.
H200용 전체 크기의 데이터 배치 생성도 확인했습니다. 상세 기록은 [VALIDATION.md](VALIDATION.md)에 있습니다.

실제 train/val이 준비된 로컬 데이터로 실행합니다. CPU는 명시적으로 축소한
crop32×64, val48×96, FP32를 사용하므로 H200 전체 해상도 성공으로 해석하지 않습니다.

```bash
PYTHONPATH=src .venv/bin/python -m ibkd_seg.cityscapes.real_smoke \
  --data-dir data/cityscapes --manifest data/cityscapes/manifest.json \
  --device cpu --cpu-small --output-dir outputs/cityscapes_real_cpu_new
```
