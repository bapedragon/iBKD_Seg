# 원인 분석 결과

- `main_l0_aggregation_checkpoint_audit_v1/`: issue 727/730의 감사된
  `main_l0_v3` iBKD checkpoint 6개에서 aggregation 가중치를 직접 읽은 관찰 결과

본실험 결과는 기존 주 결과나 이미지-loader 결과 폴더에 섞지 않고 이 경로 아래에
버전별로 추가합니다. Git에는 정리된 표와 checkpoint 해시 manifest를 두고,
checkpoint·원시 archive는 GitHub Release에 보존합니다. Smoke 결과와 feature
cache는 보존하지 않습니다.
