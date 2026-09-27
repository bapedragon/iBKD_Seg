# Cityscapes Tiny · FSKD* / C2VKD* · 2,000스텝 결과

**상태: 두 방법 모두 2,000step·전체 val500 평가 완료.** 2026-09-28 사용자 제공 로그 반영.
준비 포함 총 **5,015.55초(1시간 23분 36초)**입니다. 마지막 65,000자 로그에서
두 개별 종료 JSON과 두 결과를 모은 최종 부모 JSON을 모두 확인했습니다.

## 결과

| 방법 | mIoU (%) | Pixel accuracy (%) | 마지막 train loss | 마지막 train CE | 학습 + val 시간 |
|---|---:|---:|---:|---:|---:|
| FSKD* (DeiT-Ti recipe transfer) | 24.3009 | 80.5187 | 1.36082 | 0.536263 | 36분 58초 |
| C2VKD* (CLIP-pool) | 14.3731 | 77.1672 | 0.452723 | 1.58747 | 37분 28초 |

공통 조건은 Tiny/16·OpenMMLab DeepLabV3-R101-D8 teacher·fine train2975·crop512·batch8·
decoder1·seed1·FP32·SGD LR0.01·80k LR schedule입니다. 고정 2,000step(6번째 epoch 진행 중)의
checkpoint를 평가했고 test는 사용하지 않았습니다. β 탐색·종료 controller가 없는 고정 계수 실행입니다.
표의 loss/CE는 마지막 학습 배치 값이며 전체 val 평균 loss가 아닙니다.

- FSKD*: CE 1, logit KD 1, global 100, patch 1, attention 1,000,000.
  마지막 가중 구성항은 global 0.389570, patch 0.0195469, attention 0.0000130316,
  logit KD 0.415430으로 합계 guidance 0.824560입니다. CE를 더해 총 loss를 계산합니다.
- C2VKD*: 별도 CE 0, PDD 1, global 0.1, patch 0.1, linguistic 0.5.
  마지막 가중 구성항은 PDD 0.451963, global 0.000121009, patch 0.00000000332581,
  linguistic 0.000638952입니다. **PDD에 정답 감독이 있으므로 진단 CE를 따로 더하지 않습니다.**
  따라서 두 방법의 total loss 숫자만으로 성능을 비교할 수 없습니다.

## 확인된 실행 상태

- 두 방법의 `status=passed`, 완료/선택/저장 step 모두 2,000입니다.
- 전체 val500·19개 클래스·유효 픽셀 917,018,489개가 일치합니다.
- 로그가 보고한 초기 student/teacher 및 관측 입력 순서 검사, 고정 recipe 출처 검사는 전부 통과했습니다.
- Teacher 동결, 평가 중 student 불변, checkpoint 저장·복원 상태 비교가 통과했습니다.
  FSKD soft-rank CUDA forward/backward와 C2VKD CLIP pool 동결 검사도 통과했습니다.
- 로컬에서 개별/부모 종료 결과가 동일한지, 가중 손실 합과 total loss가 맞는지,
  19개 class IoU 평균이 mIoU와 일치하는지 콘솔 반올림 오차 내에서 확인했습니다.
- 결정성 경고는 각 2,000건입니다. 이 로그는 수치 검사와 해당 실행의 내부 검사를 보여주며
  CUDA의 bitwise 재현성을 입증하지는 않습니다. 원본 checkpoint·전체 step 파일은 첨부되지 않았습니다.

## 해석과 다음 단계

2k 시점 mIoU는 기존 LG/ALG 최상위 39.1370%, iBKD λ0.25 최상위 37.3437%,
λ0.5 최상위 37.4001%보다 낮습니다. 기존 방법은 β 후보를 여러 개 비교한 결과이고,
이번 두 방법은 고정 설정 하나씩의 초기 결과이므로 최종 성능 우위로 결론내리지 않습니다.

FSKD*는 6개 클래스, C2VKD*는 **15개 클래스의 IoU가 0**입니다. C2VKD*에서 양수인 클래스는
road/building/vegetation/car 네 개뿐입니다. 이는 클래스별 분할 성능이 아직 낮다는 뜻이며,
예측 자체가 전혀 없었다는 의미는 아닙니다. 유한 loss가 유지됐으므로 NaN/Inf 발산 실패와 구분합니다.
이 로그만으로 부족한 학습 길이와 recipe/이식 영향 중 어느 것이 원인인지 확정할 수 없습니다.

FSKD*는 공개 분류 recipe의 segmentation 이식이고 C2VKD*는 재구성한 PDD와 CLIP pool 대체를
사용합니다. 두 결과 모두 저자의 원본 Cityscapes 설정을 완전 재현한 것으로 표기하지 않습니다.
C2VKD*는 추가 CLIP 사전학습을 사용한 보조 비교군으로 유지합니다.

현재 2k 실행·평가 검사는 완료됐습니다. 다음은 기존 계획에 따른 **별도 10k 실행 구성**이며,
아직 Tiny 10k를 시작하지 않았습니다. 낮은 2k 점수만으로 비교군을 제외하거나 손실 계수를
자동 변경하지 않습니다. 계수 변경이 필요하면 별도 실험으로 명시해야 합니다.
2k→10k 재개에는 실제 checkpoint 파일 보관과 전환 검증이 필요합니다.

## 기록

- [baseline_screen_summary.json](baseline_screen_summary.json): 두 결과·19 class IoU·고정 계수·실행 상태.
- [source_manifest.json](source_manifest.json): 첨부 로그/최종 JSON SHA-256 및 출처 범위.
- 최종 통합 JSON은 표식/개행 포함 6,959 ASCII byte로 마지막 65,000자에 온전히 남았습니다.
- 요청한 실행 commit은 `b683717113d37531488871f72adbf2b540be5d6f`입니다. 앞부분이 잘려
  실제 checkout commit과 이슈 번호는 확인할 수 없습니다. 최종 로그의 방법별 `source_commit`은
  FSKD/C2VKD 원저자 코드의 출처이며 실행 저장소 checkout commit과 구분합니다.
