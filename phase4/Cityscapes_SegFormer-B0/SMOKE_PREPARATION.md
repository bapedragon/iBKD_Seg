# B0 smoke 구현 검사 기록

2026-09-23. 아래는 최초 v1 준비의 로컬 확인 기록입니다. 실제 H200 v1 로그는
[요청 819 점검](reports/h200_smoke_v1_819/RESULTS.md)에 별도로 기록했습니다.
v2 수정 후에는 단위 검사 **14개 통과·CUDA 전용 1개 skip**, 실제 가중치 CPU 연결·재실행
**7/7 통과**를 확인했습니다. 이후 [v2 H200 실행](reports/h200_smoke_v2/RESULTS.md)에서도
7개 방법 모두 학습·복원·재실행·val2 검사를 통과했습니다.

## 확인한 내용

- CIRKD teacher, NVIDIA ImageNet MiT-B0, OpenAI CLIP RN101의 실제 bytes/SHA-256을 측정하고
  strict 적재했습니다. 출처·해시는 [asset 명세](configs/b0_asset_sources_v1.json)에 있습니다.
- CIRKD 원본의 미사용 `tkinter.tix`, `pip`, `mmcv.runner` import만 제거해 현대 환경에서
  적재합니다. encoder·decoder forward는 바꾸지 않습니다. 가중치는 별도 strict loader로 받습니다.
- B0의 모든 raw block 출력, stage LayerNorm 후 출력, 마지막 attention softmax를 hook으로 얻습니다.
  frozen teacher의 layer2/3/4 특징을 연결하고, auxiliary head는 적재하지만 loss에 사용하지 않습니다.
- train 데이터는 3개 필드, val 데이터는 4개 필드를 반환하는 CIRKD 인터페이스를 처리합니다.
  labelId255 등 유효 범위 밖의 값은 공통 ignore=-1로 맞춥니다.
- CPU 합성 batch2·64×64 입력과 실제 pretrained 가중치에서 **7/7 경로**가 통과했습니다.
  각 경로에서 2 update를 수행한 뒤 1번째 update의 checkpoint를 적재해 2번째를 재실행했습니다.
  student·guidance·optimizer·controller·loss의 결과가 **bitwise 일치**했습니다.
- 모든 개별 loss에서 encoder로 gradient가 전달되고 encoder·decoder·guidance가 업데이트됩니다.
  초기 student 해시는 7개 경로가 같고, LG/ALG의 loss·gradient norm은 같습니다. Teacher는 변하지 않습니다.
- unit 검사 **10개가 통과**했습니다. 정확한 Gram chunk의 값/gradient, PDD 식의 상수 차이·동일 gradient와 ignore 처리,
  KD의 면적 정규화, controller 경계·부분 구간 재개, HF K/V 역변환, FSKD attention gradient,
  좌우 logits 결합 순서, 모든 19개 클래스의 metric, CIRKD 데이터 반환 형식을 확인했습니다.

CPU 검사의 crop64는 **연결 검사만 위한 합성 입력**입니다. FSKD spatial alignment 차원도
64에 맞추며 H200 명령의 crop512·batch16 조건과 구분합니다. CPU 손실 크기를 실제 학습의
안정성이나 β 선정 근거로 사용하지 않습니다. 원시 tensor·checkpoint·실행 결과는 Git 밖에 둡니다.

## 다시 검사하는 명령

필요한 의존성이 설치된 저장소에서 실행합니다. `torchsort==0.1.10`도 필요합니다.

```bash
PYTHONPATH=src python -m pytest -q tests/test_cityscapes_b0.py
PYTHONPATH=src python -m ibkd_seg.cityscapes.b0.assets --cache /tmp/b0_assets
python phase4/Cityscapes_SegFormer-B0/scripts/check_b0_connections_cpu.py --cache /tmp/b0_assets --output /tmp/b0_cpu_check_new
```

H200 명령은 [이슈 입력안](H200_SMOKE_ISSUE.md)을 사용합니다. H200에서는 CPU 축소 입력을
사용하는 옵션이 없고 batch16/crop512 계약이 맞지 않으면 실패합니다.

## 남아 있는 검증

H200 v1에서 실제 Cityscapes 증강·CUDA 손실/gradient·torchsort CUDA 연결은 3 update까지
확인했습니다. 재실행 비교에서 실패해 val2와 frozen 상태·peak memory 최종 요약은 남지 않았습니다.
이후 v2에서 결정적 GPU 실행·bitwise 복원·재실행 비교·val2 평가·학습 구간 메모리 확인을
완료했습니다. 상세 범위와 남은 장기 학습 검증은 최신 H200 결과 기록을 따릅니다.
3개 update의 유한한 loss는 장기 학습 안정성이나 논문 재현 성능을 보장하지 않습니다.
전체 val500, 25-batch β 후보 측정, 2k/10k 선별 및 80k 장기 runner는 다음 단계입니다.
