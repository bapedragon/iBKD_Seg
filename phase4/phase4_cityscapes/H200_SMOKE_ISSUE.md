### 사용자 ID (Username)

bapedragon

### 실행할 코드의 GitHub 링크

https://github.com/bapedragon/iBKD_Seg.git

### 코드 실행 명령어

bash phase4/phase4_cityscapes/scripts/run_deeplabv3_segmenter_smoke.sh

### 사용할 이미지 선택

pytorch/pytorch:latest

### 사용 언어 (Language)

Python

### GPU 할당량 (MIG 갯수)

7

### 실험 범위

Cityscapes 적용 전 DeepLabV3-ResNet101 → Segmenter-S/16의 Vanilla/LG/ALG/iBKD 합성 smoke입니다.
실제 Cityscapes 데이터는 준비되지 않았으며 데이터·pretrained weight를 다운로드하지 않습니다.
임의 초기 가중치와 동일 합성 입력으로 crop768×768/batch2/BF16, 각 3step 및
checkpoint strict reload·optimizer/RNG 복원·동일 update 검증을 실행합니다.
합성 1024×2048 평가에서 pixel accuracy와 mIoU의 계산 경로도 확인합니다.

점수는 실제 Cityscapes 성능이 아닙니다. 본학습 실행은 포함하지 않습니다.
방법별 실패 시 자동 batch/해상도 축소 없이 중단하고 partial 결과를 보관합니다.

결과: `/app/output/cityscapes_deeplabv3_segmenter_smoke_v1/artifacts/smoke_summary.json`

완료 marker: `[CITYSCAPES_ARCH_SMOKE_DONE] status=passed methods=4/4 scientific_result=false`

조건: https://github.com/bapedragon/iBKD_Seg/blob/main/phase4/phase4_cityscapes/DEEPLAB_SEGMENTER_SMOKE.md
