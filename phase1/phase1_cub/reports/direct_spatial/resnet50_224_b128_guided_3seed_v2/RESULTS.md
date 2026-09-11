# CUB ResNet-50/224 batch128 직접 공간정보 진단 v2 — 3 seeds

상태: **H200 issue 737·739 완료 · encoder seed 1·2·3 결합 · 독립 감사 통과**

각 part-probe 셀에서 probe seed 5개를 먼저 평균한 뒤, 아래 `±`는 독립
classification encoder seed `[1,2,3]`에 대한 sample SD로 계산했습니다. Part
PCK@0.1이 주 직접지표이고 CKA와 attention–GT는 보조지표입니다.

| 방법 | Part PCK@0.1 ↑ | 정규화 위치오차 ↓ | CKA block11 ↑ | Attention AP ↑ | Pointing ↑ | FG mass ↑ |
|---|---:|---:|---:|---:|---:|---:|
| LG | 27.510 ± 2.030% | 0.2356 ± 0.0094 | 0.4285 ± 0.0166 | 0.2269 ± 0.0678 | 0.4428 ± 0.1313 | 0.1895 ± 0.0382 |
| ALG-w20 | 21.621 ± 2.716% | 0.2725 ± 0.0194 | 0.3768 ± 0.0275 | 0.2457 ± 0.0173 | 0.4819 ± 0.0361 | 0.1898 ± 0.0093 |
| iBKD λ=0.25 | 20.109 ± 2.024% | 0.2946 ± 0.0179 | 0.3184 ± 0.0152 | 0.2816 ± 0.0546 | 0.5195 ± 0.0356 | 0.2093 ± 0.0083 |
| iBKD λ=0.5 | 15.426 ± 1.858% | 0.3407 ± 0.0344 | 0.2351 ± 0.0311 | 0.2408 ± 0.0245 | 0.4956 ± 0.0165 | 0.2059 ± 0.0131 |

## 감사

- issue 739 실행시간: 42.02분
- issue 739 encoder 8개는 감사된 issue 730 classification checkpoint의 파일·state hash와 일치
- 새 part-probe checkpoint 40개 모두 SHA-256, `weights_only=True`, strict load, 유한값 검사 통과
- issue 737의 seed-1 part-probe 20개 감사 결과를 합쳐 encoder seed 3개 × probe seed 5개 구성
- seed 2·3의 validation 후보 120개와 선택 40개를 끝낸 뒤 official test를 열었으며 test는 선택에 사용하지 않음
- 세 seed 전체 official-test 평가는 part probe 60회 + attention 12회 = 72회
- CKA는 validation 600장만 사용했고 official test를 사용하지 않음
- train의 공식 visible keypoint 64,697개 중 image 5007의 프레임 밖 1개만 사전 규칙대로 제외; validation 7,164개와 test 69,546개는 모두 유효
- issue 737의 32개와 issue 739의 64개, 총 96개 attention 정성 이미지를 두 보고서에 보존

## 해석

- **주 지표 Part PCK는 LG가 가장 높습니다.** LG는 27.510%, ALG-w20은 21.621%, iBKD λ=0.25는 20.109%, iBKD λ=0.5는 15.426%입니다.
- CKA block11도 LG가 0.4285로 가장 높고, 세 encoder seed 모두 `LG > ALG-w20 > iBKD-0.25 > iBKD-0.5` 순서입니다.
- Attention의 3-seed 평균은 AP·pointing·foreground mass 모두 iBKD λ=0.25가 가장 높습니다. 다만 seed별 1위가 달라지고, 주 Part PCK 및 CKA와 방향이 일치하지 않으므로 보조적인 국소화 신호로만 해석합니다.
- Frozen segmentation probe에서도 LG가 가장 높았던 결과와 함께 보면, 현재 CUB 프로토콜은 **iBKD가 LG/ALG보다 공간정보를 전반적으로 더 잘 보존한다는 가설을 지지하지 않습니다.**
- 이 결론은 잠긴 ResNet-50/224 scratch·현재 image-loader 조건의 guided 네 방법에 한정합니다. 이후 loader를 바꾸면 기존 결과는 그대로 보존하고 새 버전에서 모든 비교 방법을 동일 조건으로 다시 실행해야 합니다.
