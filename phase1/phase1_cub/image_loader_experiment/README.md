# CUB 이미지 loader 실험

완료된 CUB ResNet-50/224 v3 주 실험과 분리해, 학생 입력 loader의 crop·광학
증강만 바꾼 사후 실험을 한곳에서 관리합니다.

- [PROTOCOL.md](PROTOCOL.md): L0/L1/L2 정의, 선택 규칙, test 접근 규칙
- `configs/`: 학습 전 잠근 machine-readable 설정
- `scripts/`: H200 진입점
- `reports/`: smoke, 실패 감사, L0/L1/L2 결과와 해석

과거 커밋과 H200 로그가 가리키는 `phase1/phase1_cub/configs/`, `scripts/`,
`reports/loader_pilot/`, `LOADER_EXPERIMENT_PROTOCOL.md` 경로에는 호환용 symbolic
link만 남깁니다. 실험 자산의 canonical 위치는 이 폴더이며, 기존 설정 파일의
내용과 SHA-256은 변경하지 않습니다.

현재 Stage B의 세 완료 로그를 합친 표는
[L0/L1/L2 로그 결과](reports/full_v1_log_snapshot/RESULTS.md)에 있습니다. 사전
규칙상 L2가 선택됐지만 결과 archive와 checkpoint 감사가 남아 있으므로,
이 결과는 최종 논문 주장 전까지 예비 결과로 취급합니다.

선택된 L2로 guided 네 방법과 encoder seed 1만 먼저 확인하는 smoke 진입점은
다음과 같습니다.

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_l2_guided_preliminary_smoke_b128_seed1.sh
```

이 smoke는 2-epoch 진단값이며 논문 결과가 아닙니다. 분류 4개와 frozen
segmentation probe 4개가 validation 선택 뒤 official test까지 각각 한 번
통과하는지만 확인합니다.

H200 issue 753 smoke는 `classification=4/4`, `probe_candidates=12/12`,
`selected_probes=4/4`, `tasks=16/16`, `status=pass`로 완료됐습니다. 설정을 바꾸지
않은 L2 단일-encoder-seed 예비 본학습 진입점은 다음과 같습니다.

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_l2_guided_preliminary_full_b128_seed1.sh
```

본학습은 분류 4개 × 300 epoch와 frozen probe 4 encoder × 5 probe seeds ×
LR 3개 × 100 epoch를 수행합니다. validation 선택이 모두 끝난 뒤 선택된 분류·probe
checkpoint만 official test에서 한 번 평가하며, checkpoint 24개와 최종 표를
출력에 보존합니다.
