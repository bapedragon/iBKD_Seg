# Cityscapes Tiny · Vanilla 10,000step 결과

**후속 반영:** [iBKD λ0.25 후보 7의 10k 결과](../ti16_ibkd_lambda0p25_b7_10k/RESULTS.md)는
mIoU 42.1373%이며, 증류 상위 후보 8개 중 현재 3개가 10k를 완료했습니다.
아래 비교 표와 진행 상태는 Vanilla 완료 당시의 기록입니다. 최신 비교는 위 링크를 확인합니다.

**학습 10,000step과 전체 val500 평가를 정상 완료했습니다.** 2026-09-28 사용자 제공 로그 반영.
준비 포함 **7,009.04초(1시간 56분 49초)**, 학습+val은 **6,560.256초(1시간 49분 20초)**입니다.
학습 6,455.7초, 전체 val 104.556초이며 실패·시간 중단·미평가 항목이 없습니다.

## 동일 학습량 비교

| 방법 | β | Step | mIoU (%) | Pixel accuracy (%) | Vanilla 대비 mIoU (%p) |
|---|---:|---:|---:|---:|---:|
| ALG · 후보 7 | 0.11195046919685056 | 10,000 | 48.3442 | 90.2803 | +2.1667 |
| Vanilla | 0 | 10,000 | 46.1775 | 87.2847 | 기준 |
| iBKD λ0.25 · 후보 1 | 0.02461834122158468 | 10,000 | 43.8013 | 86.2812 | −2.3762 |

ALG·iBKD 수치는 [먼저 완료한 10k 실행](../ti16_alg_ibkd025_top1_10k_vanilla2k/RESULTS.md)에서 가져왔습니다.
공통 Tiny/16·decoder1·crop512·batch8·seed1·FP32·SGD LR0.01·80k poly LR schedule·fine train2975와
고정 마지막 10,000step의 전체 fine val500 평가 조건입니다. Test는 사용하지 않았습니다.

Vanilla는 teacher·guidance를 로드하지 않고 CE만 학습했습니다. 마지막 train loss=CE=**0.218122**,
guidance=0, 마지막 LR=0.00886887입니다. 27번째 epoch 진행 중인 checkpoint를 평가한 결과로,
전체 val 평균 loss 또는 27epoch을 완주한 결과가 아닙니다.
기존 Vanilla 2k mIoU 36.6126%보다 **9.5649%p 높습니다.** 기존 2k와 이번 10k는 각각 새로 시작한 실행이며,
이번 실행의 2k 중간 mIoU를 평가한 것은 아닙니다.

## 해석

- 현재 확인된 단일 seed·선택된 후보들의 10k 결과는 **ALG > Vanilla > iBKD λ0.25**입니다.
  이번 β의 iBKD는 실행에 성공했지만 Vanilla보다 mIoU가 2.3762%p 낮습니다.
- 이는 iBKD 전체가 효과 없다는 결론은 아닙니다. 아직 λ0.25의 다른 상위 후보 및 λ0.5 후보들의
  10k 결과가 없고, 80k 최종 성능과 seed 반복도 확인하지 않았습니다.
- iBKD가 7,441step에서 가이던스를 종료한 것이 성능 차이의 원인인지는 알 수 없습니다.
  종료 전후 중간 mIoU와 같은 설정에서 종료만 바꾼 비교 실행이 없습니다.
  이번 Vanilla 결과도 종료 시점의 최적성을 판정하는 실험은 아닙니다.
- ALG·iBKD는 2k에서 β 후보를 탐색한 결과이므로 후보 탐색 비용과 선택 과정을 함께 기록합니다.
  이번 결과에 맞춰 β·종료 규칙·평가 시점·exclusion rule을 자동 변경하지 않습니다.

## 확인된 실행 및 로그 상태

- 종료 상태 `passed`, 목표·완료·선택·저장 step 모두 10,000입니다.
- 전체 val500·19개 클래스·유효 픽셀 917,018,489개, metric의 유한성 및 19 IoU 평균과 mIoU의
  일치를 반올림 오차 내에서 확인했습니다. 19개 클래스 IoU는 모두 양수입니다.
- Teacher/guidance/controller 미사용, 평가 중 student 불변, checkpoint 저장·복원 상태 비교가
  로그에 정상으로 보고됐습니다. 개별/부모 종료 JSON의 결과도 동일합니다.
- 1step과 25step 간격 로그 **401개가 전부 남아 있고**, 모두 유한한 loss=CE, β=guidance=0입니다.
  이번 2,000step의 loss/CE=0.399096·gradient norm=1.43505는 이전 Vanilla 2k 마지막 값과
  콘솔 표시 정밀도에서 일치합니다. 이는 전체 학습 궤적이나 파일 해시의 동일성 검증을 대신하지 않습니다.
- 단독 실행의 `same_initial_student`·`same_observed_input_prefixes`는 해당 묶음 내부 검사입니다.
  **이전 ALG/iBKD 작업과 전체 입력·초기 상태 해시를 대조한 것은 아닙니다.**
- 결정성 경고는 10,000건, 전체 기록 경고는 10,004건입니다. 실행은 완료됐지만 GPU bitwise
  재현성을 새로 입증하지는 않습니다. 전체 step 파일과 checkpoint 원본은 첨부되지 않았습니다.
- 첨부는 65,000자 콘솔 뒷부분이며 설치 앞부분과 checkout 출력이 없습니다. 마지막 결과 JSON은
  표식·개행 포함 **3,720 ASCII byte**로 온전히 남았습니다.

## 다음 단계와 기록

이제 Vanilla·ALG 후보 7·iBKD λ0.25 후보 1의 10k 결과가 있습니다.
기존 증류 상위 후보 8개 계획에서는 여전히 2개 완료이며, LG 7·1, ALG 1, iBKD λ0.25 7,
iBKD λ0.5 1·6의 여섯 개가 남아 있습니다. 추가 학습을 자동 시작하지 않았습니다.

- [followup_summary.json](followup_summary.json): 전체 metric·19 IoU·비교 표·검증 범위.
- [source_manifest.json](source_manifest.json): 첨부/최종 JSON/config SHA-256 및 출처.
- 요청한 commit: `941627791841d2f89fc310b263b13f11f257a2bd`. 실제 checkout commit·이슈 번호는
  첨부된 로그에서 확인할 수 없어 미확인으로 기록했습니다.
- Checkpoint 경로는 요약에 기록했으며 다음 작업까지 실제 서버 파일이 보존되는지는 별도 확인이 필요합니다.
