# 원인 분석 결과

- `main_l0_aggregation_checkpoint_audit_v1/`: issue 727/730의 감사된
  `main_l0_v3` iBKD checkpoint 6개에서 aggregation 가중치를 직접 읽은 관찰 결과

향후 smoke와 full 결과는 기존 주 결과나 이미지-loader 결과 폴더에 섞지 않고 이
경로 아래에 버전별로 추가합니다. Checkpoint와 feature cache는 Git history가 아닌
H200 결과 archive/GitHub Release로 보존합니다.
