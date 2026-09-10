# CUB ResNet-50/224 seed-1 직접 공간정보 진단 v2

상태: **H200 issue 737 완료 · 독립 감사 통과 · encoder seed 1 전용**

| 방법 | Part PCK@0.1 ↑ | 정규화 위치오차 ↓ | CKA block11 ↑ | Attention AP ↑ | Pointing ↑ | FG mass ↑ |
|---|---:|---:|---:|---:|---:|---:|
| LG | 28.174 ± 0.114% | 0.2326 | 0.4420 | 0.2184 | 0.5247 | 0.1751 |
| ALG-w20 | 19.945 ± 0.059% | 0.2890 | 0.4023 | 0.2643 | 0.5178 | 0.1885 |
| iBKD λ=0.25 | 22.393 ± 0.068% | 0.2748 | 0.3341 | 0.2765 | 0.4943 | 0.1999 |
| iBKD λ=0.5 | 14.394 ± 0.074% | 0.3742 | 0.2656 | 0.2127 | 0.4767 | 0.2206 |

Part PCK가 주 직접지표이고 CKA와 attention–GT는 보조지표입니다. `±`는 동일
encoder에서 학습한 5개 part-probe seed의 sample SD이며 독립 encoder 반복이
아닙니다.

## 감사

- 실행시간: 20.54분
- encoder 4개는 issue 727의 감사된 batch128 seed-1 checkpoint hash와 일치
- part-probe checkpoint 20개 모두 SHA-256, `weights_only=True`, strict load, 유한값 검사 통과
- 원시 part-probe checkpoint와 history는 [검증된 GitHub Release](artifact_release.json)에 보존
- validation으로 20개 선택을 끝낸 뒤 official test를 열었으며 test는 선택에 사용하지 않음
- Part probe 20회 + attention 4회 = official test 평가 총 24회
- CKA는 validation 600장만 사용해 official test를 사용하지 않음
- train의 공식 visible keypoint 64,697개 중 이미지 5007의 프레임 밖 1개만 규칙대로 제외;
  validation 7,164개와 test 69,546개는 모두 유효
- 32개 attention 정성 이미지와 layerwise CKA heatmap을 함께 보존

## 해석

- **주 지표 Part PCK는 LG가 28.174%로 가장 높습니다.** iBKD λ=0.25는
  22.393%로 ALG-w20(19.945%)보다 높지만 LG보다 낮습니다.
- CKA block11도 LG가 0.4420으로 가장 높고 iBKD λ=0.25는 0.3341입니다.
  따라서 teacher spatial feature와의 유사도 관점에서도 iBKD 우위가 아닙니다.
- Attention AP는 iBKD λ=0.25가 0.2765로 가장 높고, foreground mass는
  iBKD λ=0.5가 0.2206으로 가장 높습니다. 반면 pointing은 LG가 0.5247로
  가장 높아 attention 신호는 지표별로 엇갈립니다.
- 결론적으로 이 seed-1 직접진단은 iBKD λ=0.25에 일부 attention/ALG 대비
  국소화 신호는 보여주지만, **“LG/ALG보다 공간정보를 전반적으로 더 잘
  보존한다”는 핵심 주장을 지지하지 않습니다.**
- encoder seed가 하나뿐이므로 최종 통계 결론은 아닙니다. 사전 고정된 seed 2·3
  실행과 설정을 이 결과를 보고 바꾸면 안 됩니다.
