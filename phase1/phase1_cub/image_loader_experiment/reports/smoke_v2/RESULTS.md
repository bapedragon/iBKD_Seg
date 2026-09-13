# CUB loader pilot smoke v2 결과

판정: **PASS — 본 pilot 실행 경로 사용 가능**

L0/L1/L2 × LG·ALG-w20·iBKD λ=0.25/0.5의 validation-only smoke가 모든
완료 gate를 통과했습니다. 총 12개 2-epoch 분류 checkpoint, segmentation probe
후보 36개, part probe 후보 36개, CKA 144개, attention row 12개와 정성 이미지
48개가 생성됐고 official test 접근은 0회였습니다. 전체 실행시간은 581.47초였고
본 pilot의 보수적 선형 외삽은 14시간 54분 8초입니다.

이 결과가 말해주는 것은 다음뿐입니다.

- 세 loader와 네 guided 방법 모두 분류→frozen segmentation→part→CKA→attention
  경로가 정상 작동합니다.
- 최대 관측 CUDA allocated/reserved는 각각 약 11.91 GB/15.88 GB로 OOM이
  없었습니다.
- 단일 10시간 작업에는 전체 pilot이 들어가지 않으므로 loader별 세 작업으로
  분할해야 합니다.

2-epoch 수치는 학습 초기의 실행 진단값이므로 loader·방법·lambda 선택이나 논문
해석에 사용할 수 없습니다. 본 pilot은 사전 고정한 300-epoch 분류와 5 probe
seeds × 3 LR × 100 epoch를 세 loader 모두에서 끝낸 뒤 validation Part PCK를
주 기준으로 loader를 선택합니다. 상세 수치와 출처는 [summary.json](summary.json)과
[source_manifest.json](source_manifest.json)에 보존합니다.
