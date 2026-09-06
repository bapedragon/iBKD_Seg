# Phase 1 — Oxford-IIIT Pet 공간 표현 검증

상태: **batch 64/128 분류·batch 64 frozen probe 완료 — batch 128 probe 대기**

현재 LOCK한 프로토콜과 full-run 계약은 [PROTOCOL.md](PROTOCOL.md), 기계가 읽을 수
있는 설정은
[`configs/oxford_iiit_pet_phase1_v1.json`](configs/oxford_iiit_pet_phase1_v1.json)에
있습니다.

2026-09-06에 batch 64/128의 teacher와 6설정 × encoder seed 3개를 모두
완료했습니다. 공식 test-once와 동일 초기화·teacher·validation split 계약,
각 profile checkpoint 19개의 hash·strict-load·유한값 감사를 통과했습니다.
[batch 64](reports/classification/batch64/RESULTS.md),
[batch 128](reports/classification/batch128/RESULTS.md) 분류 결과와
[profile 비교](reports/classification/BATCH_PROFILE_COMPARISON.md)를 함께 보고합니다.

Batch 64 frozen probe 본 실험도 완료했으나 iBKD λ=0.25/0.5가 matched ALG보다
각각 `-1.796/-2.358`%p 낮았고, 여섯 paired encoder-seed 차이가 모두
음수였습니다. 따라서 batch 64의 1차 가설은 지지되지 않았으며
[전체 probe 결과](reports/frozen_probe/batch64/RESULTS.md)와
[고정 정성 panel](reports/frozen_probe/batch64/QUALITATIVE.md)에 근거를 남깁니다.

## 핵심 질문

> 품종 분류 정답만으로 학습한 iBKD encoder에 동물의 위치와 형태 정보가
> Vanilla, KD, LG, ALG보다 선형적으로 더 쉽게 읽히는 상태로 남아 있는가?

[Phase 0.5](../phase0.5/README.md)는 Flowers 자동 pseudo-mask로 실행 경로만
확인했습니다. Phase 1은 Oxford-IIIT Pet의 공식 pixel-level trimap을 사용해 실제
공간정보 보존 여부를 판단하는 첫 본 실험입니다.

## 실험 흐름

Oxford-IIIT Pet에는 각 이미지의 품종 label과 pixel trimap이 함께 있습니다.
먼저 모델에는 mask를 보여주지 않고 품종 label만으로 분류 encoder를 학습합니다.
그 뒤 분류 head를 제거하고 encoder를 고정한 다음, 모든 방법에 같은 작은
segmentation probe를 붙입니다.

```text
Pet 이미지 + 품종 라벨
        ↓ 조건을 맞춘 분류 학습
Vanilla / KD / LG / ALG / iBKD encoder
        ↓ 분류 head 제거 및 encoder 고정
최종 feature [B, 192, 14, 14]
        ↓ 동일한 Conv2d(192, 2, 1) probe만 trimap으로 학습
동물 / 배경 예측 비교
```

iBKD 결과가 여러 encoder seed에서 일관되게 높다면, 분류만 학습했는데도 iBKD
feature에 위치·형태 정보가 더 선형적으로 읽기 쉬운 형태로 남았다는 근거가 됩니다.

## 고정한 full classification matrix

데이터 split, test-once, 모델 구조, metric과 probe 방식을 유지합니다. timing
결과에서 12개 조합 모두 OOM 없이 실행되고 batch 64/128의 시간이 비슷했으므로,
성능 수치를 보기 전에 다음 두 batch profile과 두 iBKD λ를 모두 실행·보고하기로
고정했습니다.

```text
(Vanilla, KD, LG, ALG, iBKD-0.25, iBKD-0.5)
× (batch 64, batch 128) × encoder seed (1, 2, 3)
= student 36 runs
```

H200 요청은 batch 64와 batch 128로 나눴습니다. 각 요청은 공통 teacher 1회와 해당
batch의 student 18회로 구성했습니다. 실제 시간은 batch 64가 7시간 29분 43초,
batch 128이 7시간 23분 29초였습니다. 결과가 좋은 batch나 λ만 골라 main 결과로
바꾸지 않고 두 profile을 별도 표로 모두 남깁니다.

## 본 실험 초안에 포함된 공통 계약

1. 공식 이미지, 품종 label, trimap과 split의 byte size·SHA-256 및 1:1 대응
2. trimap의 동물/배경 정의와 애매한 경계 픽셀 ignore 규칙
3. 공통 CNN teacher, DeiT-Tiny student와 분류 학습 schedule
4. Vanilla/KD/LG/ALG/iBKD의 방법별 고정값과 동일성 범위
5. encoder-training seed, checkpoint 선택 규칙과 test 접근 정책
6. frozen feature layer, normalization 여부와 cache 계약
7. 공통 probe 초기화, LR grid, epoch, seed와 validation 선택 규칙
8. foreground/background IoU, 2-class mIoU, Dice와 비영상 baseline

위 항목 중 batch와 λ를 제외한 초안은 2026-09-05에 기록했습니다. 공식 `trainval` 3,680장은 품종별
20장의 고정 validation을 떼어 `train/validation=2,940/740`으로 사용하고, 공식
test 3,669장은 최종 평가 전까지 선택 과정에서 사용하지 않습니다. 분류 encoder는
seed `[1,2,3]`, probe는 각 encoder마다 seed `[1,2,3,4,5]`를 사용합니다.

full-data timing의 runtime만 사용해 H200 작업을 두 개로 나눴습니다. 각 실행 시작
전에 이미지 archive SHA-256, image-label-trimap 대응, split manifest와 method
contract를 검사합니다. 분류 validation으로 checkpoint를 고른 뒤에만 official test를
각 checkpoint당 정확히 한 번 평가합니다.

## 비교가 뜻하는 것

| 비교 | 확인하는 내용 |
|---|---|
| iBKD vs Vanilla | CNN teacher guidance가 공간정보 보존에 도움이 되는가? |
| iBKD vs KD | 최종 예측 전달보다 grid를 유지한 전달이 유리한가? |
| iBKD vs LG | 학습 가능한 정렬이 고정된 공간 정렬보다 유리한가? |
| iBKD vs ALG | iBKD 추가 구조가 adaptive guidance보다 유리한가? |

가장 가까운 주 비교는 조건이 일치한 iBKD–ALG입니다. 조건이 다른 checkpoint나
pilot 결과는 별도의 탐색 결과로 표시합니다.

## 해석 범위

긍정적 결과는 iBKD feature의 **공간정보 선형 복원성**이 더 좋다는 근거입니다.
iBKD가 완성된 segmentation 모델이나 세그멘테이션 전용 KD보다 우수하다는 뜻은
아닙니다. 이후 Phase에서 공간 대조 실험, 공통 decoder fine-tuning, PASCAL VOC와
세그멘테이션 전용 KD 비교가 필요합니다.

## 계산 자원

- 로컬: Pet 데이터 감사, 단위 테스트, 결과 curation과 정성 확인
- H200: 완료한 timing·두 분류 profile·batch 64 probe, 남은 batch 128 probe

## Batch 64 frozen-probe smoke

본 probe 전에 실행 경로, frozen-feature 계약, 메모리와 시간을 확인하는
비과학적 smoke가 준비되어 있습니다. batch 64 분류 결과 중 encoder seed 1의
6개 checkpoint를 사용하고, 공식 train/validation `2,940/740` 전체에 probe seed
1과 LR `[0.01, 0.03, 0.1]`을 각각 2 epoch 실행합니다. 공식 test는 생성하거나
평가하지 않습니다.

```bash
bash phase1/scripts/run_probe_smoke_b64.sh
```

기본 입력은 `/app/output/phase1_pet_full_b64_v1`입니다. 이전 H200 출력이 그
경로에 유지되어 있으면 그대로 사용하고, 새 컨테이너라서 입력이 없으면
[batch 64 checkpoint Release manifest](reports/classification/batch64/checkpoint_release.json)에
고정된 GitHub Release asset을 자동으로 내려받습니다. 전체 student 18개와 teacher
1개 checkpoint가 들어 있으며 byte size와 SHA-256을 통과해야만 압축을 풉니다.
따라서 이전 컨테이너 mount는 필수가 아닙니다.

checkpoint binary는 저장소의 모든 clone을 영구적으로 무겁게 만들지 않도록 Git
commit이 아니라 GitHub Release asset으로 보존합니다. manifest와 다운로드·검증
코드는 Git 이력에 포함합니다. 필요하면 `PHASE1_B64_CLASSIFICATION_ROOT`로 입력
설치 경로만 바꿀 수 있습니다.

smoke의 validation IoU는 파이프라인 검사용이며 방법 선택이나 논문 결론에 사용할
수 없습니다. 결과는 `/app/output/phase1_pet_probe_b64_smoke_v1`에 작은 summary,
CSV와 smoke probe checkpoint만 저장하고, 수 GB의 feature cache는 `/app/scratch`에
둡니다.

## Batch 64 frozen-probe 본 실험

Smoke 통과 뒤 다음 명령으로 LOCK된 본 실험을 실행했고 H200 작업 706에서
완료했습니다.

```bash
bash phase1/scripts/run_probe_full_b64.sh
```

6개 설정 × encoder seed 3개 × probe seed 5개에서 LR 3개를 각각 100 epoch
학습하므로 LR 후보는 270개이고, validation으로 선택되는 probe는 90개입니다.
90개 선택과 strict reload가 모두 끝났다는
`selection_complete_before_test.json`을 먼저 기록한 다음에만 공식 test를 엽니다.
선택된 각 probe는 grid/input metric과 고정 정성 예측을 한 번의 test pass에서 함께
계산하므로 공식 test 평가는 probe당 정확히 1회입니다.

기본 경로는 다음과 같습니다.

- 분류 checkpoint 설치: `/app/scratch/phase1_pet_full_b64_v1_input`
- Pet 데이터와 임시 feature cache: `/app/scratch`
- 회수할 결과: `/app/output/phase1_pet_probe_b64_full_v1`

결과 폴더에는 90개 선택 probe, 모든 원값 CSV, 집계 JSON, 실행 상태, 두 비영상
baseline과 결과 전에 고정한 test 8장의 정성 panel이 포함됩니다. iBKD λ 0.25와
0.5를 결과로 고르지 않도록 정성 panel도 두 세트 모두 저장합니다. 수 GB의 frozen
feature cache와 데이터셋은 scratch에서 사용 후 결과 폴더에 복사하지 않습니다.

검증된 결과는 [batch 64 결과 보고서](reports/frozen_probe/batch64/RESULTS.md)에
있습니다. 전체 raw 산출물은 Git history 대신
[GitHub Release manifest](reports/frozen_probe/batch64/artifact_release.json)에
고정했습니다. 다음 단계는 protocol을 사후 변경하는 것이 아니라 batch 128
checkpoint에 같은 probe 계약을 적용하는 것입니다.
