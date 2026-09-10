# CUB-200-2011 frozen segmentation probe 결과

분류 checkpoint를 고정한 뒤 동일한 probe로 수행한 정량·정성 결과와 산출물
manifest를 이 폴더에 기록합니다.

ResNet-50/224 guided 네 방법의 batch-128/64 encoder seed 1 결과는 H200 issue
727에서 완료되어 [v4 부분 결과](resnet50_224_b128_b64_guided_seed1_v4/RESULTS.md)에
감사·정리했습니다. Issue 730의 seed 2·3을 합친
[batch-128 guided 4방법 3-seed 결과](resnet50_224_b128_guided_3seed_v5/RESULTS.md)는
encoder seed를 독립 반복 단위로 삼고, 각 encoder의 probe seed 평균을 먼저 계산한
뒤 방법별 평균과 표본표준편차를 보고합니다. LG가 가장 높았으며 iBKD λ=0.25는
ALG-w20보다도 평균 0.637%p 낮았습니다. Vanilla/KD가 없어 최종 6방법 매트릭스는
아직 미완료입니다.

H200 issue 716에서 완료된 ResNet-56/32 v2 guided probe 45개는
[구버전 보존 보고서](../legacy_resnet56_v2_guided/RESULTS.md)에 따로 정리했습니다.
이는 v3 probe 결과와 합치거나 최종 iBKD 주장 판정에 사용하지 않습니다.
