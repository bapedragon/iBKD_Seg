# Phase 0 최종 결정

상태: **완료 — Flowers GT 사용 보류 / Oxford-IIIT Pet 경로 선택**

결정일: 2026-09-02

## 결론

저장소 구조, 실행 환경, 보존된 체크포인트, 공식 파일 계약, feature 추출 경로,
metric 구현은 준비됐습니다. 하지만 Flowers-102의 공식 자동 segmentation은
semantic-segmentation 주장을 뒷받침하는 단독 ground truth로 사용하기에 적합하지
않습니다.

- **Phase 0.5는 pseudo-mask 파이프라인 진단으로만 허용합니다.** 기존 Flowers
  Ours/ALG 체크포인트를 재사용할 수 있으며, KD는 조건이 일치하지 않는 탐색
  baseline으로 표시합니다. 결과는 semantic segmentation이 아니라 “자동 마스크
  복원성”으로 표현합니다.
- **과학적 검증인 Phase 1은 Oxford-IIIT Pet trimap으로 진행합니다.** 조건을
  맞춘 Vanilla/KD/LG/ALG/iBKD encoder는 H200에서 학습합니다.
- 모델별 결과를 확인한 뒤 자동 마스크 실패 사례를 임의로 제거해서는 안 됩니다.

## 검증된 기술 계약

- 로컬 기준 환경에서 Python 3.13.1, PyTorch 2.11.0, torchvision 0.26.0,
  timm 1.0.27 import를 확인했습니다.
- Ours, ALG, 탐색용 KD 체크포인트의 byte size와 SHA-256이 manifest와
  일치합니다. 세 체크포인트 모두 metadata 검증과 DeiT-Ti strict loading을
  통과했습니다.
- 모든 encoder가 12개의 NCHW feature grid를 출력하며 최종 shape은
  `[1, 192, 14, 14]`입니다. 학습 가능한 encoder parameter는 0개입니다.
- 386-parameter `Conv2d(192, 2, 1)` probe의 backward 결과 probe gradient
  tensor는 2개, encoder gradient tensor는 0개입니다.
- 단위 테스트 14개가 로컬에서 통과했습니다.
- 공식 파일 4개는 고정된 byte size와 SHA-256에 모두 일치합니다.
- 이미지, 마스크, 라벨은 각각 8,189개입니다. 공식 train/validation/test
  1,020/1,020/6,149 split은 서로 겹치지 않으며 모든 ID를 포함합니다. 모든
  이미지–마스크 쌍이 정상적으로 decode되고 크기가 일치합니다.

## Ground-truth 품질 gate

Flowers-102 원 논문은 배포 segmentation이 반복적 color/shape 알고리즘으로 만든
자동 결과라고 설명하며, 결과가 완벽하지 않을 수 있음을 명시합니다. 배포 파일은
사람이 class index를 표시한 mask가 아니라 JPEG blue-screen composite입니다.

고정한 변환 규칙은
`composite = alpha * original + (1 - alpha) * RGB(0, 0, 255)`에서 alpha를
복원하고 `alpha >= 0.5`를 flower로 정의합니다. 원본 해상도 8,189쌍을 모두
검사한 결과는 다음과 같습니다.

- 전경이 하나도 없는 마스크 220개(2.687%): train 28, validation 36, test 156
- 배경이 0.5% 미만인 마스크 22개(0.269%): train 2, validation 5, test 15
- 전경 비율의 0/1/5/50/95/99/100% quantile:
  0.000/0.000/0.112/0.342/0.655/0.816/1.000
- 모든 split에서 원본–마스크를 직접 대조하여 두 실패 유형을 확인

전경이 전혀 없는 대표 ID는 36(train), 38(validation), 45(test)입니다. 배경이
사실상 없는 대표 ID는 1270(train), 4246(validation), 568(test)입니다.

## 다음 gate

[`GROUND_TRUTH_OPTIONS.md`](GROUND_TRUTH_OPTIONS.md)를 검토해 Oxford-IIIT Pet을
선택했습니다. 진행 순서는 다음과 같습니다.

1. Flowers Phase 0.5 로컬 파이프라인 진단 완료
2. Oxford-IIIT Pet 데이터·trimap 계약과 Phase 1 프로토콜 고정
3. 조건을 맞춘 Pet encoder와 multi-seed 학습을 H200에서 수행
4. GT 기반 probe 신호가 안정적일 때만 PASCAL VOC로 확장

재현 가능한 작은 결과 요약은 `reports/`에 보존합니다. 전체 로컬 보고서와
dataset/checkpoint 파일은 Git에 올리지 않습니다.
