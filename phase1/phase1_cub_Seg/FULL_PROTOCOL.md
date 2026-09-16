# CUB 직접 segmentation 탐색 본학습 v1

이 프로토콜은 real-data smoke 통과 뒤 Vanilla/LG/ALG/iBKD를 동일 조건으로 끝까지 학습해
확장 가능성을 확인하는 탐색 본학습입니다. 논문 수치로 확정된 실험은 아닙니다.

## 데이터와 선택

- 기존 CUB classification과 동일한 `5,394 train / 600 validation / 5,794 test`를 사용합니다.
- 공식 grayscale mask의 값이 `> 0`이면 bird, 아니면 background입니다.
- Validation 2-class mIoU가 가장 높은 epoch를 선택하고 동률이면 앞 epoch를 선택합니다.
- Official test는 validation 선택이 완료된 checkpoint에만 한 번 적용합니다.

## 공통 조건

- 입력 224×224, binary CE, class weighting 없음
- Train은 전체 이미지를 square resize한 뒤 epoch별 결정적 horizontal flip만 사용
- Evaluation은 square resize만 사용
- Batch 32, BF16, gradient clip 1.0, seed 1
- Teacher와 네 student 모두 100 epochs
- 매 epoch validation, `latest.pt`와 validation-selected `best.pt` 저장
- 중단 시 epoch boundary에서 model/optimizer/controller/RNG를 strict resume

Teacher는 scratch ResNet-50과 공통 multi-level decoder를 SGD로 학습합니다. Guided 세 방법은
같은 validation-selected teacher checkpoint를 frozen/eval 상태로 공유합니다. Student는 scratch
DeiT-Tiny/16과 같은 decoder 형식을 AdamW로 학습합니다.

방법별 규칙은 smoke와 동일합니다. LG/ALG는 block `[0, 6, 11]`, iBKD는 12 blocks 전체를
사용합니다. ALG warm-up은 0, iBKD warm-up은 20 epochs, beta는 2.5, iBKD fusion ratio는
0.25입니다.

## 결과 해석

주 지표는 dataset-level 2-class mIoU입니다. Bird IoU, background IoU, bird Dice, pixel
accuracy를 모두 함께 보고합니다. 한 seed의 탐색 실행이므로 일반화 결론이나 통계 검정에는
사용하지 않습니다.

로그의 마지막 줄에는 teacher와 네 방법의 validation/test 전체 metric, 선택 epoch, 마지막
학습 loss, guidance 종료 epoch가 JSON으로 출력됩니다. 동일 내용은 `full_summary.json`에도
저장됩니다.
