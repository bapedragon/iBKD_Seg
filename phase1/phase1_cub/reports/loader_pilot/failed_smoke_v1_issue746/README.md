# CUB loader pilot smoke v1 실패 기록 — H200 issue 746

판정: **실행 실패, 과학적 결과 아님**

Issue 746은 L0/L1/L2 × LG·ALG-w20·iBKD λ=0.25/0.5의 2-epoch 분류 12개와
임시 checkpoint 12개를 모두 생성했습니다. 동일한 학생 초기 state hash를
사용했고 OOM이나 데이터·teacher 검증 문제는 없었습니다. 그러나 첫 encoder의
train 5,394장·validation 600장 segmentation feature cache를 만든 직후 공통 probe
생성기가 요구하는 `probe.initialization` 필드를 v1 config에서 찾지 못해
`KeyError: 'initialization'`으로 중단됐습니다.

따라서 segmentation LR 후보, part probe, CKA, attention은 모두 0개이고 official
test도 열지 않았습니다. 로그의 2-epoch validation accuracy는 실행 진단값일 뿐
loader·방법·lambda 비교나 선택에 사용할 수 없습니다.

후속 `loader_pilot_smoke_v2`는 방법, 데이터 split, metric, 선택 규칙을 그대로
유지하면서 공통 probe runtime에 필요한 초기화·parameter count·optimizer·scheduler·
loss schema만 완성합니다. 새 컨테이너에서 v2 전체 경로를 다시 수행하며 v1
checkpoint를 과학적 결과에 합치지 않습니다.

기계 판독 가능한 완료 수와 출처 hash는 [failure_summary.json](failure_summary.json)에
기록했습니다. 원본 H200 로그와 임시 checkpoint는 Git에 넣지 않습니다.
