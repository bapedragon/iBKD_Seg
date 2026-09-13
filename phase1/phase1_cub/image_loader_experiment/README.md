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
Stage C 본학습 전에는 새 L2 smoke만 실행할 수 있습니다.

선택된 L2로 guided 네 방법과 encoder seed 1만 먼저 확인하는 smoke 진입점은
다음과 같습니다.

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_l2_guided_preliminary_smoke_b128_seed1.sh
```

이 smoke는 2-epoch 진단값이며 논문 결과가 아닙니다. 분류 4개와 frozen
segmentation probe 4개가 validation 선택 뒤 official test까지 각각 한 번
통과하는지만 확인합니다.
