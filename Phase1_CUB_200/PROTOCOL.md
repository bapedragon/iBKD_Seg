# Phase 1 CUB-200-2011 프로토콜

상태: **초안 체크리스트 — LOCK 전**

이 문서는 결과나 본실험 코드를 만들기 전에 CUB-200-2011 전용 실험 계약을
확정하기 위한 자리입니다. 아래 항목이 모두 결정되고 machine-readable config의
해시가 기록되기 전에는 H200 본실험을 시작하지 않습니다.

## 고정할 항목

- 공식 이미지·분류 annotation·segmentation mask의 출처, byte size, SHA-256
- 이미지 ID와 class label, 공식 split, mask의 1:1 대응 및 누락 처리
- train/validation/test 분할과 test-once 정책
- pixel class 정의, resize/interpolation, 경계 또는 무효 픽셀 처리
- CNN teacher와 ViT student 구조 및 초기화
- Vanilla, KD, LG, ALG, iBKD의 공통 설정과 방법별 고정값
- batch size, epoch, optimizer, learning-rate schedule과 seed
- 분류 checkpoint의 validation-only 선택 기준
- frozen feature 위치·shape·normalization과 encoder freeze 검사
- 공통 probe 구조, LR 후보, epoch, seed와 validation 선택 기준
- IoU, mIoU, Dice 등 최종 metric과 비영상 baseline
- smoke와 본실험의 분리, 산출물 manifest와 결과 보고 형식

## Pet과의 관계

Pet은 완료된 독립 실험이므로 CUB 결과에 맞춰 Pet의 LOCK config나 결과를
수정하지 않습니다. 가능한 조건은 Pet과 맞추되, CUB 데이터 특성 때문에 달라지는
항목은 근거와 함께 이 문서에 명시합니다.
