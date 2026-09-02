# Phase 0 — 기초 검증 및 프로토콜 확정

## 목적

세그멘테이션 head를 학습하기 전에 공식 데이터, 보존된 분류 체크포인트,
feature 추출 경로, 평가 metric을 신뢰할 수 있는지 확인합니다.

로컬과 H200의 역할은 [COMPUTE_PLAN.md](COMPUTE_PLAN.md), 실제 GT 선택지는
[GROUND_TRUTH_OPTIONS.md](GROUND_TRUTH_OPTIONS.md)에 정리되어 있습니다.

## 입력 자산

- Oxford Flowers-102 공식 이미지
- Oxford Flowers-102 공식 자동 segmentation
- `imagelabels.mat`, `setid.mat`
- SHA-256으로 고정한 Ours/ALG 조건 일치 체크포인트
- SHA-256으로 고정한 탐색용 KD 체크포인트

## Step 0.1 — 실행 환경

기준 dependency는 `requirements.txt`에 기록합니다. PyTorch, torchvision, timm,
NumPy, Pillow, SciPy를 import할 수 있는 호환 환경이 필요합니다.

## Step 0.2 — 체크포인트 감사

`results/Ours`, `results/ALG`, `results/KD`가 들어 있는 기존 실험 폴더를
`--source-root`로 지정합니다.

```bash
PYTHONPATH=src python -m ibkd_seg.phase0.checkpoints \
  --manifest manifests/checkpoints.json \
  --source-root /path/to/IBAM_KD_H200_V2 \
  --output phase0/reports/checkpoint_audit.local.json
```

Deep audit는 byte size, SHA-256, metadata, strict `state_dict` loading, 12개의
NCHW intermediate feature, 최종 `[1, 192, 14, 14]` grid, 완전히 frozen된
encoder를 확인합니다. 또한 386-parameter Phase 1 probe를 실제로 backward하여
probe에만 gradient가 생기는지 검사합니다.

## Step 0.3 — 공식 데이터 다운로드 및 압축 해제

압축 상태의 전체 다운로드 크기는 약 548MB입니다.

```bash
bash phase0/scripts/download_flowers102.sh
```

기본 저장 위치는 Git에서 제외된 `data/flowers102/`입니다. 공식 서버가 느릴 수
있으므로 downloader는 기존 파일의 끝에서 재개하고 기본 8개의 검증된 HTTP byte
range를 병렬로 받습니다. 연결 수는 `IBKD_SEG_DOWNLOAD_CONNECTIONS`로 조정할 수
있습니다.

## Step 0.4 — 데이터셋 감사

```bash
PYTHONPATH=src python -m ibkd_seg.phase0.flowers_data \
  --data-root data/flowers102 \
  --manifest manifests/flowers102.json \
  --output phase0/reports/dataset_audit.local.json
```

다음 항목을 검사합니다.

- 공식 archive의 byte size와 SHA-256
- 이미지·마스크 개수와 1:1 ID 대응
- canonical ID 범위
- 공식 split의 개수, 상호 배타성, 전체 ID 포함 여부
- label 개수와 class ID
- 모든 이미지–마스크의 decode 및 원본 해상도 일치 여부

공식 segmentation은 class-index PNG가 아니라 JPEG blue-screen composite입니다.
더 중요한 점은 원 논문이 이를 자동 flower-segmentation 알고리즘의 출력으로
설명한다는 것입니다. 따라서 완전한 human semantic-segmentation ground truth로
표현해서는 안 됩니다.

고정 변환식은 다음과 같습니다.

```text
composite = alpha * original + (1 - alpha) * RGB(0, 0, 255)
foreground = alpha >= 0.5
```

감사 코드는 일부 sample의 transition pixel과 threshold 민감도를 기록하는 한편,
전체 8,189개 마스크를 원본 해상도로 검사하여 empty, near-empty, near-full
결과를 찾습니다.

공식 파일에 대해 이 명령은 의도적으로 종료 코드 1을 반환합니다. 구조 검사는
모두 통과하지만 전경이 없는 마스크 220개와 배경이 0.5% 미만인 마스크 22개가
존재하기 때문입니다. 이는 다운로드나 decoder 오류가 아니라 연구 gate 결과입니다.
작은 재현 요약은 `reports/dataset_audit.summary.json`, 전체 ID 목록은 Git에서
제외되는 로컬 보고서에 저장됩니다.

## Step 0.5 — 단위 테스트

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

데이터 준비 후 Phase 0 감사를 한 번에 다시 실행할 수도 있습니다.

```bash
IBKD_SEG_PYTHON=/path/to/python \
  bash phase0/scripts/run_phase0_audits.sh \
  /path/to/IBAM_KD_H200_V2 \
  data/flowers102
```

## 완료 체크리스트

- [x] Dependency 계약 기록 및 import 확인
- [x] 모든 체크포인트의 byte size와 SHA-256 일치
- [x] 모든 체크포인트 metadata 일치
- [x] 세 DeiT-Ti `state_dict` strict loading 통과
- [x] 최종 frozen feature shape `B x 192 x 14 x 14` 확인
- [x] 공식 파일 다운로드 및 hash 고정
- [x] 이미지–마스크 8,189쌍 확인
- [x] 공식 train/validation/test ID의 상호 배타성과 전체 포함 확인
- [x] 모든 이미지–마스크 크기 일치
- [x] 실제 파일에 기반한 binary mask 변환 규칙 확정
- [x] 자동/pseudo-mask라는 사실과 실제 GT 선택지 문서화
- [x] 세그멘테이션 metric 단위 테스트 통과
- [x] Phase 1A 입력 경로와 알려진 leakage 주의사항 문서화

Phase 0의 최종 판단은 **보류(HOLD) / 프로토콜 수정**입니다. 과학적 검증인
Phase 1B 전에 `GROUND_TRUTH_OPTIONS.md`의 한 경로를 확정해야 하며,
Oxford-IIIT Pet trimap을 권장합니다.
