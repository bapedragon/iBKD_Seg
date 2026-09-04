# iBKD 세그멘테이션 확장성 검증

이 저장소는 데이터가 부족한 이미지 분류 환경에서 iBKD가 학습한 공간
표현이 세그멘테이션과 같은 dense prediction 문제에도 유효한지 단계적으로
검증합니다.

실험은 Phase별 gate 방식으로 진행합니다. 현재 Phase의 결과와 판단 근거가
문서로 확정된 뒤에만 다음 Phase로 넘어갑니다.

## 현재 상태

**Phase 1A 완료 — Phase 1B Pet 데이터·학습 계약 준비 중.**

코드, 체크포인트, 공식 파일, split, feature shape, metric 구현은 정상입니다.
하지만 전체 데이터 감사에서 전경이 없는 마스크 220개와 배경이 사실상 없는
마스크 22개를 확인했습니다. 따라서 Phase 1A에서는 파이프라인 진단용으로만
Flowers 자동 마스크를 사용하고, 실제 확장성 판단인 Phase 1B에서는 신뢰할 수
있는 pixel-level ground truth를 사용해야 합니다. 2026-09-04에 전체 공식 split
Phase 1A frozen-probe 실행과 정성 확인이 통과했으며, Flowers 방법 순위는
진단값으로만 보존했습니다.

| Phase | 핵심 질문 | 상태 |
|---|---|---|
| 0 | 입력 데이터와 평가 계약을 신뢰할 수 있는가? | 감사 완료 / 보류 |
| 1 | 고정된 iBKD feature에서 dense mask가 더 잘 복원되는가? | 1A 통과 / 1B 준비 |
| 2 | 관측된 차이가 공간적이고 여러 seed에서 재현되는가? | 대기 |
| 3 | 더 강한 공통 decoder와 fine-tuning에서도 차이가 유지되는가? | 대기 |
| 4 | multi-class semantic segmentation으로 일반화되는가? | 대기 |
| 5 | dense task 전용 iBKD 확장이 필요한가? | 대기 |

전체 단계와 gate 조건은 [ROADMAP.md](ROADMAP.md)에 정리되어 있습니다.
Phase 1A와 Pet Phase 1B의 목적 및 수행 과정은
[phase1/README.md](phase1/README.md)에서 쉽게 확인할 수 있습니다.
완료한 Phase 1A의 [결정문](phase1/PHASE1A_DECISION.md),
[정량 JSON](phase1/reports/phase1a_summary.json),
[실제 Flowers 정성 결과](phase1/reports/PHASE1A_QUALITATIVE.md)도 Git에 함께
보존합니다.

## 저장소 구성 원칙

- `phase0/`, `phase1/`처럼 Phase 단위의 최상위 폴더를 사용합니다.
- 각 Phase 폴더에는 해당 단계의 절차, 명령어, 보고서, 결정문을 둡니다.
- 여러 Phase가 공유하는 구현은 `src/`에 둡니다.
- 데이터셋, 체크포인트, 원시 실행 결과는 Git에 올리지 않습니다.
- 사용하는 체크포인트와 공식 데이터 파일은 SHA-256으로 식별합니다.
- 논문 표의 수치와 로컬 재현 체크포인트의 수치를 구분하여 기록합니다.
- README, 로드맵, 결정문 등 사용자용 문서는 한국어로 작성합니다.

## Phase 0 빠른 시작

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

보존된 Flowers-102 체크포인트 감사:

```bash
PYTHONPATH=src python -m ibkd_seg.phase0.checkpoints \
  --manifest manifests/checkpoints.json \
  --source-root /path/to/IBAM_KD_H200_V2 \
  --output phase0/reports/checkpoint_audit.local.json
```

데이터 준비와 전체 감사 절차는 [phase0/README.md](phase0/README.md)에
정리되어 있습니다. 로컬과 H200의 역할 분리는
[phase0/COMPUTE_PLAN.md](phase0/COMPUTE_PLAN.md), 최종 gate 판단은
[phase0/DECISION.md](phase0/DECISION.md)에서 확인할 수 있습니다.
