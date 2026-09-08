# CUB-200-2011 분류 결과

최종 ResNet-50/224 scratch v3의 공용 Teacher 본학습은 H200 issue 722에서
완료됐습니다. 선택 epoch, validation·official-test 결과, 실행 환경, 전체 history,
split·protocol snapshot과 checkpoint 독립 감사는
[resnet50_224_teacher_v3](resnet50_224_teacher_v3/RESULTS.md)에 기록했습니다.

Teacher checkpoint는 GitHub Release에 보존하고 파일 SHA-256과 model-state
SHA-256을 함께 고정합니다. 앞으로 Vanilla, KD, LG, ALG-w20, iBKD λ=0.25,
iBKD λ=0.5의 6설정×3seed 분류 결과가 완료되면 같은 v3 하위 보고서로 추가합니다.

ResNet-56/32 v2의 ALG/iBKD 분류 결과는 최종 v3 결과가 아니므로 이 폴더에
병합하지 않고 [별도 보존 폴더](../legacy_resnet56_v2_guided/RESULTS.md)에 둡니다.
