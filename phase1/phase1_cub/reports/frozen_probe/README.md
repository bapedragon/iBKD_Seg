# CUB-200-2011 frozen segmentation probe 결과

분류 checkpoint를 고정한 뒤 동일한 probe로 수행한 정량·정성 결과와 산출물
manifest를 이 폴더에 기록합니다.

최종 ResNet-50/224 v3의 6방법×3 encoder seed probe는 아직 실행 전입니다.
완료되면 encoder seed를 독립 반복 단위로 삼고, 각 encoder의 probe seed 평균을
먼저 계산한 뒤 방법별 평균과 표본표준편차를 보고합니다.

H200 issue 716에서 완료된 ResNet-56/32 v2 guided probe 45개는
[구버전 보존 보고서](../legacy_resnet56_v2_guided/RESULTS.md)에 따로 정리했습니다.
이는 v3 probe 결과와 합치거나 최종 iBKD 주장 판정에 사용하지 않습니다.
