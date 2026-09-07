# Phase 1 — CUB-200-2011 공간 표현 검증

상태: **실험 폴더 생성 — 프로토콜 미고정·실험 미실행**

이 폴더는 Oxford-IIIT Pet 결과와 섞이지 않도록 CUB-200-2011 독립 반복만
관리합니다. 완료된 Pet 실험과 결과는 [phase1_pet](../phase1_pet/README.md)에
그대로 보존합니다.

## 핵심 질문

> CUB-200-2011의 종 분류 label만으로 학습한 iBKD encoder가 Vanilla, KD, LG,
> ALG보다 새의 위치·형태 정보를 더 선형적으로 복원 가능한 형태로 보존하는가?

## 예정한 실험 흐름

1. CUB-200-2011 분류 label로 조건을 맞춘 encoder를 학습합니다.
2. 분류 validation만 사용해 각 encoder checkpoint를 선택합니다.
3. 선택한 encoder를 완전히 고정합니다.
4. 모든 방법에 동일한 segmentation probe를 붙여 pixel mask로 학습합니다.
5. validation으로 probe를 선택한 뒤 test를 최종 평가합니다.

Pet과 질문 및 비교 원칙은 같지만, 데이터 split과 mask 출처를 포함한 CUB용
프로토콜은 별도로 고정합니다. Pet 설정 파일을 복사해 곧바로 본실험에 사용하지
않습니다.

## 디렉터리

- `configs/`: 결과를 보기 전에 잠근 CUB 전용 machine-readable 설정
- `scripts/`: 데이터 감사, smoke, 본실험 및 결과 정리 진입점
- `reports/classification/`: 200종 분류 결과와 checkpoint manifest
- `reports/frozen_probe/`: frozen segmentation probe 정량·정성 결과
- `results/raw/`: 로컬 원시 산출물 전용 경로이며 Git에는 포함하지 않음

프로토콜 확정 전 체크 항목은 [PROTOCOL.md](PROTOCOL.md)에 기록합니다.

## 현재 주의사항

- CUB-200-2011 본체의 분류 annotation과 별도 segmentation mask archive의
  이미지 ID 대응, 누락, 해시를 먼저 감사해야 합니다.
- train/validation/test, ambiguous·배경 픽셀 처리, 입력 해상도, teacher,
  batch size, 방법별 hyperparameter, seed와 checkpoint 선택 규칙은 아직
  확정하지 않았습니다.
- 데이터셋, checkpoint, feature cache와 원시 H200 결과는 Git에 올리지 않습니다.
  검증된 작은 요약·manifest·정성 예시만 `reports/`에 반영합니다.
