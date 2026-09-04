# Phase 0.5 — Flowers pseudo-mask 파이프라인 진단

상태: **완료 — 파이프라인 통과**

## 이 단계를 따로 둔 이유

Flowers-102이 제공하는 segmentation은 사람이 직접 표시한 pixel ground truth가
아니라 자동 알고리즘이 만든 blue-screen composite입니다. 전체 감사에서 빈 mask와
거의 완전 전경인 mask가 확인됐기 때문에, Flowers 결과로 iBKD의 segmentation
확장성을 주장하지 않습니다.

Phase 0.5의 질문은 하나입니다.

> 기존 분류 checkpoint를 고정한 상태에서 데이터 처리, feature cache, 선형
> segmentation probe 학습, metric과 정성 시각화가 끝까지 정상 작동하는가?

과학적 비교는 공식 trimap을 사용하는 [Phase 1 Pet 실험](../phase1/README.md)에서
진행합니다.

## 실제 실행 흐름

```text
Flowers 이미지 + 자동 pseudo-mask
              ↓
Ours / ALG / 탐색용 KD 분류 encoder 고정
              ↓ 최종 feature [192, 14, 14]
동일한 Conv2d(192, 2, 1) probe만 학습
              ↓
전경/배경 예측과 IoU·Dice 계산
```

Probe는 bias를 포함해 386개 parameter만 가집니다. encoder는 `eval()` 상태로
고정되고 gradient가 생기지 않습니다. 정확한 계약은 [PROTOCOL.md](PROTOCOL.md),
machine-readable 설정은
[`configs/flowers102_phase05_v1.json`](configs/flowers102_phase05_v1.json)에 있습니다.

## 실행

먼저 16/16/16 표본의 2-epoch smoke test를 실행합니다.

```bash
bash phase0.5/scripts/run_smoke.sh /path/to/IBAM_KD_H200_V2
```

통과 후 Ours/ALG와 탐색용 KD의 전체 공식 split을 실행합니다.

```bash
bash phase0.5/scripts/run_full.sh /path/to/IBAM_KD_H200_V2
```

feature와 target cache는 `phase0.5/results/raw/cache/`, probe와 원시 예측은
`phase0.5/results/runs/`, 전체 로컬 JSON은 `phase0.5/reports/*.local.json`에
저장되며 Git에서 제외됩니다.

## 보존한 결과

- 해석과 최종 gate: [DECISION.md](DECISION.md)
- 소형 정량 요약: [`reports/summary.json`](reports/summary.json)
- 실제 입력–pseudo-mask–예측: [reports/QUALITATIVE.md](reports/QUALITATIVE.md)
- 그림 provenance: [`reports/figures/flowers102/manifest.json`](reports/figures/flowers102/manifest.json)
- 구현: [`src/ibkd_seg/phase05/`](../src/ibkd_seg/phase05/)
- 단위 테스트: `tests/test_phase05_*.py`

## 명명 이력

이 실험은 폴더를 분리하기 전에 내부적으로 `Phase 1A`라는 이름으로 실행됐습니다.
따라서 이미 고정·실행된 config의 `protocol_id`와 결과 JSON에는
`flowers102_phase1a_...` 문자열이 남아 있습니다. 이는 기존 config SHA-256과
cache 호환성을 깨지 않기 위한 **불변 실행 식별자**일 뿐이며, 현재 연구 단계는
Phase 0.5입니다.
