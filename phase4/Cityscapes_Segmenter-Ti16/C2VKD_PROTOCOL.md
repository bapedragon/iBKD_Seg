# Tiny C2VKD* (CLIP-pool) smoke 프로토콜

2026-09-26 사용자가 CLIP-pool 대체 비교군까지 포함하도록 요청하여 v3에 추가했습니다.
저자 C2VKD의 완전 재현이 아닌 **공개 논문·코드 기반 재구성과 Tiny 이식**입니다.
기존 B0 C2VKD* 손실·pool 구현을 재사용하며 B0 실험이나 L/16 결과를 수정하지 않습니다.

## 출처와 대체 범위

- [C2VKD 논문](https://arxiv.org/html/2310.07265v1),
  [공개 코드](https://github.com/zhengxuJosh/C2VKD/tree/fe1ab3d6f815969058451c221229a744fe872a47).
- 공개 코드가 요구하는 `pretrain_model0.pth` 대신
  [OpenAI CLIP RN101](https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt)의
  `visual.attnpool`만 사용합니다. CLIP CNN이나 text encoder를 입력 이미지에 실행하지 않습니다.
- 파일 291,791,292 bytes, SHA-256
  `8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599`를 먼저 확인합니다.
  공식 Tiny/teacher 캐시의 `weights/clip_rn101.pt`에 별도로 보관합니다.
- pool은 2,048 입력 채널·32 attention heads·512 출력 채널입니다. 원본 7×7 spatial
  positional embedding을 16×16으로 bilinear 보간하고 global 위치는 유지합니다.
  pool은 항상 eval, requires_grad=False이며 시작/종료 state SHA 일치를 확인합니다.
- 동일 구조를 사용한다는 사실이 저자의 pooling 가중치 또는 teacher feature 분포와
  동일하다는 뜻은 아닙니다. **CLIP 사전학습 정보가 추가**되어 별도 보조 비교군으로 기록합니다.
- 공개 소스에 `utils.kd_losses.get_dkd_loss` 구현이 없어 PDD는 논문 식 8/9에 따라
  재구성했습니다. 이 부분도 저자가 검증한 구현이라는 주장을 하지 않습니다.

## 공통 모델 및 손실 연결

Teacher는 기존 OpenMMLab DeepLabV3-R101-D8 Cityscapes checkpoint를 유지합니다.
원 논문의 DeepLabV3+ teacher·별도 학습 설정을 그대로 재현하는 실험이 아닙니다.
Student는 다른 6경로와 동일한 초기 Segmenter-Ti/16, decoder1, crop512, batch8,
seed1, FP32, SGD 및 80k horizon입니다. 데이터·증강·평가도 공통입니다.

- Student: encoder 마지막 LayerNorm의 실제 출력에서 CLS 하나를 제외한 patch feature.
  192채널, crop512에서 32×32입니다. Hook은 출력이나 encoder forward를 바꾸지 않습니다.
- Teacher: 동결 ResNet layer4의 2,048채널 feature.
- Visual: student Conv1×1(192→2048, bias 없음), teacher와 함께 32×32에 정렬합니다.
- Linguistic: student GAP→Conv1×1(192→512), teacher feature는 동결 pool을 통과합니다.
- Pixel logits: 공통 Segmenter 규칙인 bilinear align_corners=False로 label 해상도에
  맞추고 ignore255 픽셀을 제외합니다.

고정 목적함수는 `PDD + 0.1×global + 0.1×patch + 0.5×linguistic`입니다.
**별도 CE를 더하지 않습니다.** PDD에 정답 label이 들어가며 CE는 비교 진단용으로만 표시합니다.
β 후보 탐색이나 guidance 종료 controller는 사용하지 않습니다.

- PDD: target/non-target 두 확률로 합치고 `(teacher + ground-truth)/2`를 정규화합니다.
  student→target KL에 0.5를 적용하고 유효 픽셀 평균을 취합니다. T=1입니다.
- Global/linguistic: 공개 코드처럼 teacher→student KL, 모든 원소 평균입니다.
  논문 식의 방향과 코드의 차이는 기존 B0 구성과 동일하게 코드를 따릅니다.
- Patch: batch 전체의 spatial token을 펼친 후 채널 L2 정규화와 row centering을 하고
  모든 token 쌍의 Gram MSE를 계산합니다. 분모2047, chunk256은 정확 계산을 분할하는
  메모리 최적화이며 쌍을 샘플링하거나 제거하지 않습니다.

## 검사 및 해석

25개 초기 batch 손실을 측정하고 전체 상태를 복원한 뒤 25 update합니다.
네 raw loss와 가중 loss를 저장하며 encoder·decoder·visual/linguistic adapter의
비영 gradient, teacher와 pool의 무변화, 24→25번째 update 재개를 검사합니다.
최종 JSON은 C2VKD*를 `supplementary_extra_pretraining`으로 분리합니다.

FSKD*/C2VKD*의 `beta=null`은 증류가 꺼졌다는 뜻이 아니라 고정 계수 사용을 뜻합니다.
C2VKD*의 guidance/CE 비율은 **전체 PDD 포함 목적함수 대 진단 CE** 비율이므로
LG/ALG/iBKD의 추가 guidance/CE 비율과 같은 의미로 해석하지 않습니다.
val 2장 mIoU·accuracy는 연결 진단값입니다. 장기 성능과 안정성을 입증하지 않습니다.
