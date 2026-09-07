# CUB-200-2011 분류 결과

200종 분류 본실험이 완료되고 산출물 감사를 통과하면 설정별·seed별 결과와
checkpoint manifest를 이 폴더에 기록합니다.

첫 guided shard 원본에는 teacher checkpoint 1개와 student best-validation
checkpoint 9개가 포함됩니다. Teacher의 파일·model-state hash를 고정한 뒤 두 번째
baseline shard가 이를 그대로 사용하며, 최종적으로 18개 student 결과를 병합합니다.
