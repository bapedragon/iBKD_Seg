# Phase 1 — CUB-200-2011 공간 표현 검증

상태: **batch 128 통합 smoke 준비 — 본실험 프로토콜 미고정·본실험 미실행**

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

## Batch 128 분류 → frozen probe 통합 smoke

다음 비과학적 smoke는 CUB 다운로드와 mask 대응, 고정 validation split, 분류
checkpoint 저장·strict load, encoder freeze, feature cache 및 probe 학습 경로를 한
번에 점검합니다.

```bash
bash phase1/phase1_cub/scripts/run_combined_smoke_b128.sh
```

공식 train 5,994장 중 클래스별 3장, 총 600장을 seed 2027 validation으로
분리하고 나머지 5,394장을 학습에 사용합니다. 공식 test 5,794장은 smoke에서
열거나 평가하지 않습니다. scratch ResNet-56 teacher 1개와 batch-128
DeiT-Tiny의 Vanilla, KD, LG, ALG-w20, iBKD λ=0.25/0.5를 seed 1에서 각각 2 epoch
실행합니다. 이어서 각 smoke encoder를 완전히 동결하고 LR
`[0.01, 0.03, 0.1]`의 probe를 각각 2 epoch 실행합니다.

Pet에서 이미 확인한 ALG의 epoch-2 조기 종료를 CUB에서 다시 주 비교 조건으로
사용하지 않도록, CUB의 유일한 ALG 설정은 controller 종료 판정 warm-up 20으로
사전 고정합니다. 이는 원본 ALG와 구분해 모든 로그와 표에서 `ALG-w20`으로 씁니다.

기본 결과 경로는 `/app/output/phase1_cub_b128_combined_smoke_v2`, 데이터와 큰
feature cache는 `/app/scratch`입니다. 마지막 로그에는 여섯 분류 validation
진단값, 여섯 probe validation 진단값, 선형 시간 외삽과 `25/25` 작업 완료 여부가
출력됩니다. 이 수치로 방법·lambda·checkpoint를 선택하거나 논문 결과를 주장할 수
없습니다. smoke 통과 후에도 [PROTOCOL.md](PROTOCOL.md)의 본실험 계약을 별도로
LOCK해야 합니다. Smoke는 encoder seed 1만 사용하고 본실험 분류는 encoder seed
`[1, 2, 3]`을 모두 실행합니다.

## 현재 주의사항

- 통합 smoke가 CUB-200-2011 본체와 별도 segmentation archive의 train/validation
  이미지·mask 대응과 공식 MD5를 먼저 검사합니다. 본실험 전에는 전체 데이터의
  byte size·SHA-256까지 별도 LOCK해야 합니다.
- train/validation/test, ambiguous·배경 픽셀 처리, 입력 해상도, teacher,
  batch size, 방법별 hyperparameter, seed와 checkpoint 선택 규칙은 아직
  확정하지 않았습니다.
- 데이터셋, checkpoint, feature cache와 원시 H200 결과는 Git에 올리지 않습니다.
  검증된 작은 요약·manifest·정성 예시만 `reports/`에 반영합니다.
