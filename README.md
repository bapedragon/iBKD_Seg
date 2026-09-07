# iBKD 세그멘테이션 확장성 검증

이 저장소는 데이터가 부족한 이미지 분류 환경에서 iBKD가 학습한 공간 표현이
세그멘테이션과 같은 dense prediction 문제에도 유효한지 단계적으로 검증합니다.

실험은 Phase별 gate 방식으로 진행합니다. 현재 Phase의 결과와 판단 근거가 문서로
확정된 뒤에만 다음 Phase로 넘어갑니다.

## 현재 상태

**Phase 0.5 완료 — Phase 1 전체 실행·감사 완료, 핵심 iBKD > LG/ALG 가설 No-Go.**

Phase 0에서 코드·체크포인트·공식 파일·split·feature shape·metric 구현을
감사했습니다. Flowers 자동 마스크에는 전경이 없는 사례 220개와 배경이 사실상
없는 사례 22개가 있어, 이를 이용한 실행은 Phase 0.5 파이프라인 진단으로
분리했습니다. 2026-09-04에 전체 공식 split의 frozen-probe 실행과 정성 확인이
통과했습니다.

실제 공간정보 보존 여부는 공식 pixel-level trimap을 사용하는 Phase 1
Oxford-IIIT Pet 실험에서 판단합니다.

Phase 1은 공식 test를 selection에 사용하지 않는 `2,940/740/3,669` split을
사용합니다. 12-way full-data 2-epoch timing이 모두 성공했으며, 결과를 사후
선택하지 않도록 student batch `64/128`과 iBKD λ `0.25/0.5`를 모두 3 seed로
실행하고 별도 profile로 보고합니다. 2026-09-06에 batch 64/128 분류 36개 student와
각 profile의 teacher 실행을 모두 완료했습니다. 두 frozen probe의 180개 선택
checkpoint도 독립 감사를 통과했습니다. Batch 64에서는 iBKD가 matched ALG보다
낮았고, batch 128에서만 iBKD가 ALG보다 높았지만 ALG가 세 seed 모두 epoch 2에
guidance를 종료했습니다. 종료 판정 warm-up만 20 epoch로 바꾼 사후 진단에서 ALG
mIoU가 `63.378 → 80.947%`로 회복되어 iBKD 두 λ보다 높았습니다. LG는 두
canonical profile 모두 1위였습니다. 따라서 Phase 1 핵심 가설은 No-Go이고 원래
전제의 Phase 2 진입은 보류합니다.

| Phase | 핵심 질문 | 상태 |
|---|---|---|
| 0 | 입력 데이터와 평가 계약을 신뢰할 수 있는가? | 감사 완료 |
| 0.5 | Flowers pseudo-mask로 전체 probe 파이프라인이 작동하는가? | 완료 / 통과 |
| 1 | Pet GT에서 iBKD feature의 공간정보가 더 잘 복원되는가? | 완료 / 핵심 가설 No-Go |
| 2 | 관측된 차이가 공간적이고 여러 seed에서 재현되는가? | 진입 보류 — Phase 1 전제 미충족 |
| 3 | 더 강한 공통 decoder와 fine-tuning에서도 차이가 유지되는가? | 대기 |
| 4 | multi-class semantic segmentation으로 일반화되는가? | 대기 |
| 5 | dense task 전용 iBKD 확장이 필요한가? | 대기 |

전체 단계와 gate 조건은 [ROADMAP.md](ROADMAP.md)에 있습니다.

- Flowers 점검: [phase0.5/README.md](phase0.5/README.md)
- Flowers 정량·정성 결과: [phase0.5/DECISION.md](phase0.5/DECISION.md),
  [실제 이미지](phase0.5/reports/QUALITATIVE.md)
- Pet 본 실험: [phase1/README.md](phase1/README.md)
- Pet LOCK 프로토콜·full-run 계약: [phase1/PROTOCOL.md](phase1/PROTOCOL.md)
- Pet batch 64 분류 결과: [phase1/reports/classification/batch64/RESULTS.md](phase1/reports/classification/batch64/RESULTS.md)
- Pet batch 128 분류 결과: [phase1/reports/classification/batch128/RESULTS.md](phase1/reports/classification/batch128/RESULTS.md)
- Pet batch 64 frozen probe 결과: [phase1/reports/frozen_probe/batch64/RESULTS.md](phase1/reports/frozen_probe/batch64/RESULTS.md)
- Pet batch 128 frozen probe 결과: [phase1/reports/frozen_probe/batch128/RESULTS.md](phase1/reports/frozen_probe/batch128/RESULTS.md)
- ALG controller warm-up 20 사후 진단: [phase1/reports/diagnostics/alg_controller_warmup20_b128/RESULTS.md](phase1/reports/diagnostics/alg_controller_warmup20_b128/RESULTS.md)
- Phase 1 결정: [phase1/DECISION.md](phase1/DECISION.md)

## 저장소 구성 원칙

- `phase0/`, `phase0.5/`, `phase1/`처럼 연구 단계별 최상위 폴더를 사용합니다.
- 각 Phase 폴더에는 해당 단계의 절차, 명령어, 보고서, 결정문을 둡니다.
- 여러 Phase가 공유하는 구현은 `src/`에 둡니다.
- 데이터셋, 체크포인트, feature cache와 원시 실행 결과는 Git에 올리지 않습니다.
- 사용하는 체크포인트와 공식 데이터 파일은 SHA-256으로 식별합니다.
- 논문 표의 수치와 로컬 재현 체크포인트의 수치를 구분하여 기록합니다.
- README, 로드맵, 결정문 등 사용자용 문서는 한국어로 작성합니다.

## 빠른 시작

기준 환경은 Python 3.10 이상, PyTorch 2.11.0, torchvision 0.26.0,
timm 1.0.27입니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

단위 테스트 실행:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Phase 0 데이터 준비와 감사는 [phase0/README.md](phase0/README.md), 완료한
Flowers probe 재현 명령은 [phase0.5/README.md](phase0.5/README.md)에 있습니다.
