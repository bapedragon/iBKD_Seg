# H200 2k v2 pack3 결과

2026-09-24 사용자 제출 로그를 점검했습니다. **iBKD λ=0.5 네 후보 모두 2,000 step을
완료**했고 전체 val500, checkpoint 복원·재실행, 500-step 안정성 검사를 통과했습니다.
원본 group selection과 별도 재계산이 일치하며 **β=0.4303·1.00403**을 10k에 유지합니다.

## 로그 식별과 확인 범위

- 원본: 65,004 bytes, SHA-256 `d3d1a6b25e503afb396272540823ec78371a2698af23ff48d997537a68a26446`.
- 실행 commit: `a51fe4c1c3ba03582d0a12d6fca2b478eb2f2dec`.
- 프로토콜: `cityscapes_b0_screen2000_v2`, pack3, completed 4/4.
- [점검 JSON](log_audit.json), [선별 JSON](selection.json).

첨부 시작 부분의 worker summary는 잘렸지만 **마지막 group JSON은 완전**합니다.
네 후보의 status·완료 step·loss·best/last metrics와 group selection을 직접 읽었습니다.
이번 표에는 잘린 후보 값을 다른 기록에서 가져오거나 추론해 채운 항목이 없습니다.

Best/last 평가 행렬 8개와 마지막 후보 β=4.303의 중간 평가 행렬 5개를 합해
13개 confusion matrix의 class IoU·mIoU·pixel accuracy를 재계산했습니다.
최대 절대 오차는 `1.11e-16`이고, GT 클래스별 픽셀 수와 전체 유효 픽셀
917,018,489가 모두 일치합니다. 해당 마지막 후보의 source signature는 현재 학습 코드
해시 및 고정 프로토콜·grid 해시와 일치합니다.
나머지 세 후보의 전체 중간 평가 이력과 실제 checkpoint 파일은 첨부에 없어 직접 검사하지 않았습니다.

Data verification과 32-batch 입력 preflight는 pack1 기록과 같습니다.
4개 모두 replay passed, stability500 passed, calibration input passed, teacher frozen 확인,
last_eval_step=2000, error=null, test_used=false입니다. 마지막 배치 해시도 pack1과 같습니다.
마지막 LR은 80k poly schedule의 step2000 값과 일치합니다.

## 후보별 결과

단위는 mIoU %. 네 후보 모두 best step=2000이며 best와 last mIoU가 같습니다.

| 초기 β | best/last mIoU | guidance 종료 판정 step | 마지막 CE | 다음 단계 |
|---|---:|---:|---:|---|
| **0.4303** | **45.6926** | 558 | 0.543473 | 10k 유지 1순위 |
| **1.00403** | **45.3704** | 558 | 0.558771 | 10k 유지 2순위 |
| 2.1515 | 45.1444 | 558 | 0.560284 | 2k 선별 종료 |
| 4.303 | 45.2237 | 558 | 0.531691 | 2k 선별 종료 |

순위는 반올림 전 best val mIoU로 계산했습니다. 마지막 batch의 CE가 더 낮다는 이유로
β=4.303을 선택하지 않습니다. 그 값은 validation 성능을 대체하지 않습니다.

이전 pack1 Vanilla best mIoU 45.2942% 대비 이번 최고값은 **+0.3984 percentage point**입니다.
iBKD λ=0.25의 최고 45.4024%보다도 수치상 높지만 모두 seed1의 2k 선별 결과입니다.
두 λ를 별도 비교 조건으로 유지하며 2k 결과로 하나를 제외하거나 80k 최종 우위를 주장하지 않습니다.

## Guidance 종료 확인

4개 모두 guidance warm-up=0, window50, threshold−0.02입니다.
기록된 구간 손실로 controller를 다시 계산했고 종료 epoch·미분·평활값·beta history가 일치했습니다.
3번째 관측 시 평활 변화량은 각각 약 −0.018988, −0.018657, −0.018040, −0.017648로,
모두 iBKD 종료 조건 `> -0.02`를 만족했습니다.

따라서 **3×186=558 step까지 CE+증류**, **559~2000 step은 CE만 사용**했습니다.
종료 후 1,442 step을 추가 학습한 것이며 학습 자체가 558에서 중단된 것이 아닙니다.
마지막 네 loss가 CE 하나이고 raw_guidance=null, weighted_guidance=0, beta=0,
guidance gradient norm=0인 것은 정상적인 종료 상태입니다.
10k 재개에서도 저장된 controller 상태를 유지해 guidance를 다시 켜지 않습니다.

## 시간과 전체 2k 선별 결과

설치·공통 준비를 포함한 전체 작업 시간은 **3시간 26분 21초**입니다.
각 후보는 약 49~50분이며 마지막 후보 β=4.303의 val mIoU 이력은
step400/800/1200/1600/2000에서 30.6155/37.7165/39.5863/42.1579/45.2237%입니다.

기존 [pack1](../h200_screen2000_v2_pack1/RESULTS.md),
[pack2](../h200_screen2000_v2_pack2/RESULTS.md)와 합친 다음 단계 대상은 아래와 같습니다.

| 조건 | 10k까지 유지할 설정 |
|---|---|
| Vanilla | 고정 baseline 1개 |
| FSKD* | 고정 재구현 1개 |
| LG | β=0.197479·0.460784 |
| ALG | β=0.197479·0.460784 |
| iBKD λ=0.25 | β=0.903048·3.87021 |
| iBKD λ=0.5 | β=0.4303·1.00403 |

총 **10개 실행 궤적**을 step2000의 전체 상태에서 10k까지 연장하는 단계입니다.
Pack2의 group header 및 ALG 첫 후보 앞부분이 누락됐다는 한계는 그대로 남으며,
실제 재개 준비 때 원본 group_summary와 모든 checkpoint 파일의 식별자를 확인해야 합니다.
Pack3도 `/app/output/cityscapes_b0_screen2000_v2/pack3/` 출력 묶음 전체를 보존합니다.
이번 점검은 결과 문서화이며 학습 코드·고정 설정을 변경하거나 10k GPU 작업을 시작하지 않았습니다.
