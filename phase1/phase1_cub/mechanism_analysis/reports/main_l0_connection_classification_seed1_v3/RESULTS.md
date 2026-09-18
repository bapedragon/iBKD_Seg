# main-L0 iBKD 레이어 연결 4방식 분류 비교 — seed 1

상태: **PASS — 분류 4/4, validation 선택 4/4, official test 4/4,
checkpoint 4/4**

## 목적과 고정 조건

`main_l0_v3`에서 관찰된 iBKD 분류 성능에 block→teacher-stage 연결 방식이
기여했는지 확인한 사후 ablation입니다. 네 조건은 모두 CUB-200-2011,
main-L0 loader, scratch ResNet-50/224 teacher, DeiT-Tiny/16, batch 128,
encoder seed 1, iBKD λ=0.25, 300 epoch를 사용했습니다. Guidance는 네 조건 모두
epoch 1–123에만 beta 2.5를 적용했습니다.

바뀐 값은 연결 방식 하나뿐입니다. Frozen segmentation probe는 수행하지 않았습니다.
각 방법의 checkpoint는 validation macro Top-1으로 선택했고, 네 선택이 모두 끝난
뒤 official test를 checkpoint당 한 번씩 평가했습니다. Test는 선택에 사용하지
않았습니다.

## 결과

| 연결 방식 | 선택 epoch | Validation macro Top-1 (%) | Test macro Top-1 (%) | Test overall Top-1 (%) | Test Top-5 (%) |
|---|---:|---:|---:|---:|---:|
| **learned_all** | 252 | **28.1667** | **26.3370** | **26.0614** | **51.7950** |
| fixed_last | 291 | 25.5000 | 23.7932 | 23.6279 | 49.3614 |
| fixed_stage_match | 253 | 24.8333 | 21.4168 | 21.2634 | 45.6334 |
| fixed_uniform_all | 256 | 17.1667 | 16.6497 | 16.5516 | 38.4191 |

`learned_all`의 test macro Top-1 차이는 다음과 같습니다.

- `fixed_last` 대비 **+2.5439%p**
- `fixed_stage_match` 대비 **+4.9203%p**
- `fixed_uniform_all` 대비 **+9.6873%p**

Validation과 test의 순위는 모두
`learned_all > fixed_last > fixed_stage_match > fixed_uniform_all`로 같았습니다.

## 해석

이 seed-1 통제 비교에서는 **학습형 다중 레이어 연결이 네 방식 중 분류 성능이
가장 높았습니다.** 마지막 block만 연결한 `fixed_last`보다도 test macro Top-1이
2.54%p 높아, 단순히 마지막 block guidance만으로 같은 분류 성능을 설명하기는
어렵습니다. 균등 다중 레이어 연결은 가장 낮았으므로, 여러 레이어를 연결한다는
사실만으로는 충분하지 않고 연결 가중치를 학습하는 것이 도움이 됐다는 방향을
지지합니다.

다만 이는 **seed 1 하나의 classification-only 사후분석**입니다. 따라서 이 결과만으로
일반적인 인과 효과, 공간정보 보존 또는 segmentation 우위를 주장하지 않습니다.
연구 방향을 처음부터 segmentation을 학습하는 경로로 전환했으므로 seed 2·3이나
frozen probe로 확장하지 않고, 이 결과를 Phase 1 분류 경로의 마지막 참고 결과로
보존합니다.

## 실행 감사

- 완료 marker: `status=pass`
- 새 classification checkpoint: `4/4`
- frozen probe: `0`
- 총 경과 시간: `10h 45m 43s`
- 공식 test가 checkpoint 선택에 사용됨: `false`
- 네 조건의 guidance 종료 epoch: 모두 `123`
- 원본 결과 archive/checkpoint는 제공되지 않았으며, 이 보고서는 사용자 제공 H200
  완료 로그의 최종 summary를 기준으로 작성했습니다.

수치 원본은 [summary.json](summary.json), 표 형식은
[classification_results.csv](classification_results.csv), 입력 로그 감사 정보는
[source_manifest.json](source_manifest.json)에 기록했습니다.
