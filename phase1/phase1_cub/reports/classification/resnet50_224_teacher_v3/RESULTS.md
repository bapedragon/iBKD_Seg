# CUB ResNet-50/224 scratch Teacher v3 결과

상태: **H200 issue 722 완료 · 독립 감사 통과 · 후속 v3 공용 Teacher로 고정**

## 프로토콜

- TorchVision ResNet-50, `weights=None`, external pretraining 없음
- 입력 224×224, batch 128, seed 1, 200 epoch
- CUB train/validation/official test: 5,394 / 600 / 5,794
- validation macro top-1 최대 checkpoint 선택, 동률이면 이른 epoch
- 선택 완료 및 새 모델 strict reload 뒤 official test 정확히 1회

## 결과

| 항목 | 값 |
|---|---:|
| 선택 epoch | 165 |
| 선택 epoch train top-1 | 94.866% |
| validation top-1 / macro top-1 | 42.667% / 42.667% |
| official test top-1 / macro top-1 | 41.111% / 41.178% |
| official test top-5 | 66.327% |
| validation−test macro gap | 1.489%p |
| 총 실행시간 | 25.70분 |
| peak allocated / reserved | 11.377 / 15.731 GB |

## 판정

Teacher 학습, validation-only 선택, checkpoint strict reload, official-test 1회 규칙이
모두 지켜졌습니다. Checkpoint 파일 SHA-256은 `ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3`, model-state
SHA-256은 `96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7`입니다. 이 두 hash를 후속 LG, ALG-w20, iBKD 실행에서
동시에 검사해 정확히 같은 Teacher 하나를 공유해야 합니다.

Scratch 소규모 fine-grained 분류라 선택 epoch의 train top-1과 validation/test 사이
간격이 큽니다. 이는 pretrained 결과와 비교할 수 없으며, 현재 결과만으로 iBKD의
공간정보 보존 성능을 판단할 수도 없습니다. 이 결과가 확정하는 것은 **v3 Teacher
artifact가 정상적으로 준비됐다**는 점까지입니다.

원본 checkpoint와 로그는 Git 이력이 아닌 ignored raw 경로와
[검증된 GitHub Release](checkpoint_release.json)에 보존하고, 이 폴더에는 검증
가능한 소형 결과와 hash만 추적합니다.
