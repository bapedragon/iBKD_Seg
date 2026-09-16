# H200 이슈 입력안 — Cityscapes ZIP 업로드 확인

이슈 제목: `[Request]: Cityscapes ZIP 업로드 확인 (기존 Chaoyang 폴더)`

[공식 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)에
아래 값을 입력합니다. 사용자 요청에 따라 이 이슈는 자동 등록하지 않습니다.

### 사용자 ID (Username)

bapedragon

### 실행할 코드의 GitHub 링크

https://github.com/bapedragon/iBKD_Seg.git

### 코드 실행 명령어

python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py --search-root /app/data/chaoyang

### 사용할 이미지 선택

pytorch/pytorch:latest

### 사용 언어 (Language)

Python

### GPU 할당량 (MIG 갯수)

1

### 점검 범위

운영진이 기존 차오양 데이터 폴더에 압축 상태로 올렸다고 안내한 Cityscapes ZIP 두 개의
접근 경로·파일 크기·SHA-256·전체 CRC·train/val/test 파일 수·이미지와 labelIds의 ID 대응을 점검합니다.
기존 차오양 본실험 기록에 있는 컨테이너 경로 `/app/data/chaoyang` 아래 최대 4단계만 검색합니다.
2026-09-14 로컬 검증을 마친 사용자 ZIP의 byte size와 SHA-256을 기준으로 대조합니다.

Python 표준 라이브러리만 사용하므로 패키지 설치나 데이터 다운로드가 없습니다.
압축을 풀거나 데이터 파일을 수정하지 않으며, 모델·학습·추론을 실행하지 않습니다.
GPU 계산은 없으며 요청 양식의 최소 할당량 1을 사용합니다.
Test는 ZIP 파일 목록과 CRC만 점검하며 성능 평가에 사용하지 않습니다.

결과: `/app/output/cityscapes_upload_check_v1/summary.json`.
로그에도 전체 요약 JSON을 출력하므로 운영진의 별도 결과 파일 전달 없이 판정할 수 있습니다.
통과 표식: `[CITYSCAPES_UPLOAD_CHECK_DONE] status=passed`.
파일 누락, 중복 위치, 읽기 실패, hash 불일치 또는 CRC 오류는 실패로 보고하고 종료합니다.

### 사전 검증

2026-09-16 로컬 원본 ZIP 두 개를 하위 폴더에서 찾는 형태로 실행하여 통과했습니다.
크기·SHA-256 일치, ZIP 전체 CRC, train 2,975 / val 500 / test 1,525개 및
이미지·labelIds ID 대응을 확인했습니다. 해시 불일치, 중복 경로, CRC 손상,
마운트 경로 누락, ZIP 누락도 실패로 처리하는지 별도 fixture로 확인했습니다.
이는 점검 코드의 로컬 검증 결과입니다. H200 업로드 상태는 이 이슈 실행 후 판정합니다.
