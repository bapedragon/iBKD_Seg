# Ground truth 경로 결정

상태: **Oxford-IIIT Pet 선택 — 과학적 검증인 Phase 1에 적용**

## 별도 결정이 필요한 이유

Flowers-102 분류 원 논문의 Section 2는 각 이미지를 반복적 color/shape 방법으로
자동 분할한다고 설명하며 초기 segmentation이 완벽하지 않을 수 있다고
명시합니다.

- 원 논문: https://www.robots.ox.ac.uk/~men/papers/nilsback_icvgip08.pdf

따라서 배포된 `segmim_*.jpg`는 완전한 human ground truth가 아니라 알고리즘이
만든 blue-screen 결과입니다. 원본 해상도 전체 감사에서 전경이 전혀 없는
마스크 220개와 배경이 0.5% 미만인 마스크 22개를 확인했으며, 두 실패 유형 모두
공식 train/validation/test split에 존재합니다.

## 선택지 A — Oxford-IIIT Pet trimap 사용(선택됨)

37개 반려동물 품종으로 조건을 맞춘 data-scarce 분류 encoder를 학습한 뒤, 공식
pixel-level trimap에 동일한 frozen spatial probe를 적용합니다.

- 공식 페이지: https://www.robots.ox.ac.uk/~vgg/data/pets/
- 장점: 모든 이미지에 품종 label과 실제 pixel-level trimap이 제공됩니다.
- 장점: 데이터 규모와 fine-grained 분류 특성이 기존 iBKD 환경과 가깝습니다.
- 비용: Vanilla/ALG/iBKD encoder를 동일 조건으로 새로 학습해야 하며 H200 사용을
  권장합니다.

## 선택지 B — 사람이 검수한 Flowers subset

기존 Flowers Ours/ALG 체크포인트를 유지하되, held-out mask subset을 사람이 직접
검수하고 수정합니다.

- 장점: 현재의 조건 일치 체크포인트를 그대로 사용할 수 있습니다.
- 비용: annotation과 품질 관리가 필요하고 subset 규모가 강한 논문 주장을
  뒷받침하기에 작을 수 있습니다.
- 필수 조건: 자동 마스크를 초기값으로 사용할 수는 있지만, 포함되는 모든
  마스크를 수정하고 독립적으로 재검수해야 합니다.

## 선택지 C — Flowers 자동 마스크 진단만 수행

기존 frozen probe를 자동 마스크에 적용하여 코드 경로와 과거 전처리 결과에 대한
공간 표현 복원성만 확인합니다.

- 장점: 비용이 가장 낮고 기존 Ours/ALG/KD 체크포인트를 즉시 재사용할 수
  있습니다.
- 한계: ground-truth semantic-segmentation 평가라고 표현할 수 없습니다.
- 한계: 모델 결과를 본 뒤 빈 마스크나 저품질 마스크를 임의로 제외하면 안 되며,
  사전에 정한 규칙과 함께 전체 현황을 보고해야 합니다.

## 선택지 D — 표준 semantic segmentation으로 바로 이동

PASCAL VOC 같은 benchmark로 곧바로 이동하여 조건을 맞춘 distillation 기반
세그멘테이션 모델을 학습합니다.

- 장점: 외적 타당성이 가장 높습니다.
- 비용: 저비용 representation probe를 건너뛰며 H200 실험과 방법 통합 비용이
  크게 증가합니다.

## 권장 순서

1. Flowers 전체 파일·마스크 품질 감사를 Phase 0 근거로 유지
2. Flowers 자동 마스크는 Phase 0.5 파이프라인 진단에만 사용
3. Oxford-IIIT Pet을 Phase 1 주 frozen-representation 검증 데이터셋으로 사용
4. GT 기반 probe에서 안정적인 신호가 나온 뒤 PASCAL VOC로 확장
