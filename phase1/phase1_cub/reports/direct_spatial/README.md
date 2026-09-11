# CUB-200-2011 직접 공간정보 진단 결과

분류 checkpoint에서 part landmark, teacher spatial feature, attention–GT 정렬을
직접 측정한 결과를 보존합니다. H200 issue 737의 seed 1과 issue 739의 seed 2·3을
합친 [3-seed v2 최종 보고서](resnet50_224_b128_guided_3seed_v2/RESULTS.md)에
독립 encoder seed 평균과 sample SD를 정리했습니다.

Part PCK@0.1이 주 지표이며 spatial CKA와 attention 지표는 보조 지표입니다.
[seed-1 전용 보고서](resnet50_224_b128_seed1_v2/RESULTS.md)도 원래 shard의
감사 근거와 해석 범위를 보존하기 위해 그대로 유지합니다.
