# CUB-200-2011 결과 보고서

검증된 요약, manifest와 작은 정성 예시만 이 폴더에 보존합니다. 데이터셋,
checkpoint, feature cache와 원시 실행 결과는 Git에 포함하지 않습니다.

현재 최종 프로토콜은 ResNet-50/224 scratch v3입니다. H200 issue 722의 공용
Teacher 결과와 checkpoint hash는
[분류 보고서](classification/resnet50_224_teacher_v3/RESULTS.md)에 고정했습니다.
후속 v3 학생·probe는 이 Teacher의 파일 hash와 model-state hash를 모두 확인한 뒤
실행합니다. H200 issue 727의 guided 네 방법 batch-128/64 seed-1 분류·probe는
[v4 부분 결과 보고서](frozen_probe/resnet50_224_b128_b64_guided_seed1_v4/RESULTS.md)에
정리했습니다. Issue 730의 batch-128 seed 2·3을 합친
[guided 4방법 3-seed 결과](frozen_probe/resnet50_224_b128_guided_3seed_v5/RESULTS.md)도
감사를 완료했습니다. Guided 블록은 완성됐지만 Vanilla/KD가 없어 아직 6방법×3
encoder seed 전체 매트릭스는 아닙니다.
48개 checkpoint와 전체 원시 로그는 Git history가 아닌 보고서에 연결한 검증된
[Release 자산](frozen_probe/resnet50_224_b128_b64_guided_seed1_v4/artifact_release.json)으로
보존합니다.

H200 issue 737의 seed 1과 issue 739의 seed 2·3 part localization, spatial CKA,
attention–GT 결과는
[3-seed 직접 공간정보 진단 보고서](direct_spatial/resnet50_224_b128_guided_3seed_v2/RESULTS.md)에
정리했습니다. Part PCK와 CKA는 LG가 가장 높고 attention 3개 지표 평균은 iBKD
λ=0.25가 가장 높았습니다. 주 직접지표와 보조 attention 지표가 엇갈리므로 현재
결과는 iBKD의 전반적 공간정보 우위를 지지하지 않습니다. Seed-1 전용 보고서도
원래 shard 기록으로 별도 보존합니다.

H200 issue 716의 ResNet-56/32 v2 guided 결과는 제출 시점의 계약대로 완결됐지만,
v3 확정 전의 구버전입니다. 결과와 55개 checkpoint는
[v2 보존 보고서](legacy_resnet56_v2_guided/RESULTS.md)에 별도로 보존하며 v3
6방법 비교에는 합치지 않습니다. 큰 checkpoint 묶음은 Git 객체가 아닌 각 보고서에
연결된 GitHub Release 자산으로 관리합니다.
