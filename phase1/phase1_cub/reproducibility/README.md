# CUB 재현성 진단

`main_l0_v3` iBKD λ=0.25 seed 1의 같은-seed 결과 변동을 확인한 완료된 사후
진단입니다. 입력·RNG·환경 통제 gate를 통과했으며 레이어 연결 분석은 별도의
matched-duration 프로토콜로 재개했습니다.

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

본실험은 동일 조건의 300-epoch A/B를 독립 process로 순차 실행했습니다. 각 실행은
validation macro Top-1으로 checkpoint를 선택한 뒤 strict reload하고 official test를
정확히 한 번 평가했습니다. test는 epoch나 설정 선택에 쓰지 않았습니다.

| 실행 | 선택 epoch | controller 종료 epoch | Test macro Top-1 |
|---|---:|---:|---:|
| A | 259 | 109 | 24.7611% |
| B | 162 | 133 | 25.5098% |

Test 차이는 `0.7487%p`로 두 실행은 같은 성능 범위였습니다. 다만 선택 epoch와
controller 종료 epoch, model state가 같다는 뜻은 아니며 CUDA 비결정론 때문에
bitwise 동일성도 주장하지 않습니다. 핵심 결론은 issue 760의 `10.4096%` 저성능이
A/A 양쪽에서 재현되지 않았다는 것입니다.

iBKD의 원래 CUDA 경로에는 결정론 구현이 없는 backward 연산 세 가지가 있으므로
완전 bitwise 결정론을 주장하지 않습니다. Smoke와 본실험 모두 사후 재현성 진단이며
기존 main 결과를 교체하지 않습니다. Git에는 smoke 산출물을 보존하지 않고 본실험의
요약만 남깁니다. 이 A/A는 성능 순위를 정하는 실험이 아니므로 checkpoint를 후속
방법 비교에 재사용하지 않습니다.
