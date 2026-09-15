# CUB 재현성 진단

현재 단계는 `main_l0_v3` iBKD λ=0.25 seed 1의 같은-seed 결과 변동을 먼저
해결하기 위한 사후 진단입니다. 레이어 연결 ablation의 후속 결론은 이 gate가
안정될 때까지 보류합니다.

H200 smoke 진입점:

```bash
bash phase1/phase1_cub/reproducibility/scripts/run_main_l0_ibkd_controlled_aa_smoke_b128_seed1.sh
```

Smoke는 동일한 2-epoch full-data 실행을 독립 process 두 번 수행하고 입력·RNG
제어와 실제 수치 차이를 자동 비교합니다. H200 issue 765에서 두 실행이 완료되고
실행 제어 gate `27/27`이 통과했습니다. 관측된 최대 지표 절대차는
`1.81652254128e-7`이었습니다.

본실험 진입점:

```bash
bash phase1/phase1_cub/reproducibility/scripts/run_main_l0_ibkd_controlled_aa_full_b128_seed1.sh
```

본실험은 동일 조건의 300-epoch A/B를 독립 process로 순차 실행합니다. 각 실행은
validation macro Top-1으로 checkpoint를 선택한 뒤 strict reload하고 official test를
정확히 한 번 평가합니다. test는 epoch나 설정 선택에 쓰지 않습니다. 입력·RNG hash는
epoch마다 기록하고, 최종 분류 성능·선택 epoch·controller 종료 epoch의 A/B 차이를
임의 허용오차 없이 그대로 보고합니다.

iBKD의 원래 CUDA 경로에는 결정론 구현이 없는 backward 연산 세 가지가 있으므로
완전 bitwise 결정론을 주장하지 않습니다. Smoke와 본실험 모두 사후 재현성 진단이며
기존 main 결과를 교체하지 않습니다. Git에는 smoke 산출물을 보존하지 않고, 본실험
결과를 전달받은 뒤 결과 요약과 필요한 checkpoint만 정리합니다.
