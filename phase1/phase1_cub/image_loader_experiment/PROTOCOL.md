# CUB 이미지 loader 실험 프로토콜

## 목적

강한 image augmentation이 CUB의 작은 새와 part를 잘라 공간정보 비교를 왜곡하는지
확인합니다. Loader만 바꾸고 Teacher, 학생 구조, 방법, seed, optimizer, probe와
평가 규칙은 동일하게 유지합니다.

## Loader 정의

- `L0 current strong`: 기존 main v3의 강한 crop·광학 augmentation
- `L1 weak`: crop 범위는 유지하고 강한 광학 augmentation을 제거
- `L2 conservative spatial`: crop을 더 보수적으로 바꿔 객체·part 보존을 강화

정확한 transform 값은 `cub_loader_profiles.py`와 이 폴더의 full config에
machine-readable 형태로 고정합니다.

## Stage A — 학습 전 손상 감사

동일 이미지에 각 loader를 반복 적용해 visible part 유지율, foreground 유지율,
bbox 보존을 측정합니다. 결과는
[damage audit](reports/damage_audit_v1/RESULTS.md)에 있습니다.

## Stage B — validation-only loader 본실험

- Guided 네 방법: LG, ALG-w20, iBKD λ=0.25, iBKD λ=0.5
- Encoder seed 1, batch 128, 300 epoch
- 각 encoder를 freeze하고 segmentation probe, part probe, CKA, attention 진단 수행
- official test는 사용하지 않음
- 사전 선택 주 지표: 네 방법의 validation Part PCK 평균
- 정확한 동률일 때만 segmentation mIoU 평균을 tie-break로 사용
- 분류, CKA와 attention은 loader 선택 지표가 아님

Stage B 결과는 [본실험 표](reports/full_v1_log_snapshot/RESULTS.md)에 있으며 L2가
선택됐습니다.

이 protocol은 Stage B의 loader 영향 관찰에서 종료합니다. 선택된 L2로 분류 encoder를
다시 학습하고 frozen decoder/probe를 교체하는 후속 예비실험은 정식 근거로 유지하지
않습니다. 다음 단계는 pixel mask를 처음부터 사용하는 직접 segmentation입니다.

## 보존과 해석 제한

- L0/L1/L2 간 비교를 `main_l0_v3`와 합산하지 않음
- 현재 결과는 로그 기반이며 archive·checkpoint 감사 완료 전 hash 수준 확정으로
  간주하지 않음
- Git에는 protocol, 작은 결과표와 출처 manifest만 보존
- 데이터셋, feature cache와 큰 checkpoint는 Git에서 제외
- 완료된 본학습의 선행 실행 점검 산출물은 제거
