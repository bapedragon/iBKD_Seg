# L/16 본학습 코드 검증 — 2026-09-16

이 기록은 본학습 실행 코드의 검증입니다. H200216epoch 성능 결과가 아닙니다.

## 자동 검사

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_cityscapes*.py'
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_phase1_timing.py'
bash -n phase4/phase4_cityscapes/scripts/run_cityscapes_official_l16_full.sh
```

Cityscapes23개 + 기존 controller/timing8개, 총31개 통과했습니다.
추가한 검사는2975표본의 중간 batch 재개·마지막 batch7·표본 중복/누락·모델 RNG 비간섭,
원본 평가 epoch1/5/…/216, 정확도 선택과 동률 처리, checkpoint 체크섬·직전 세대 복구,
외부 bundle best 가중치 복사 및 다른 method/identity 재개 거부를 포함합니다.

## 실제 원본 L/16 및 공개 가중치 실행

네 방법 모두 원본334,845,966 parameter student와 공개 초기 가중치를 사용했습니다.
Guided 방법은 공개 MMSeg teacher도 로드했습니다.
CPU 검증에서만 train4장·val2장·crop32·batch2·2epoch로 축소했습니다.
Optimizer schedule 분모80,352는 유지했습니다. CUDA에서는 축소 옵션을 허용하지 않습니다.

- Vanilla/LG/ALG/iBKD 모두2epoch/4update 및 원본 sliding inference·best 저장 완료.
- 네 방법의 초기 student state SHA-256, 데이터 identity, epoch별 첫 증강 batch SHA-256 일치.
- iBKD는 연속4update 실행과, 1update 후 정지 → 재개해2update 후 평가 직전 정지 →
  재개해 완료한 실행을 비교했습니다.
- 최종 student/guidance/optimizer state 해시, scheduler, controller 전체 state,
  epoch loss·validation 이력, 선택된 best checkpoint의 bytes/SHA-256·점수 모두 **정확히 일치**했습니다.
  Controller의 첫 epoch 관측이 재개 때문에 중복되지 않았습니다.

예시 검증 명령:

```bash
PYTHONPATH=src .venv/bin/python -m ibkd_seg.cityscapes.official_full --cache-root /path/to/upstream --data-dir data/cityscapes --manifest data/cityscapes/manifest.json --output-dir /tmp/l16_cpu_check --method ibkd --device cpu --cpu-small --max-steps 1
PYTHONPATH=src .venv/bin/python -m ibkd_seg.cityscapes.official_full --cache-root /path/to/upstream --data-dir data/cityscapes --manifest data/cityscapes/manifest.json --output-dir /tmp/l16_cpu_check --method ibkd --device cpu --cpu-small --resume /tmp/l16_cpu_check/resume.json
```

실제 로컬 결과 경로는 `/private/tmp/cityscapes_full_check_v1`입니다.
임시 데이터·가중치·실행 결과는 Git에 포함하지 않습니다.

## 해석의 한계

작은 입력 검사에서 LG/ALG loss는 약19.7 → 276 → 47,781로 증가했고,
네 번째 update를 포함한 두 번째 epoch 평균은 약1.87×10^19였습니다.
iBKD도 두 번째 epoch 평균 loss가 약261로 증가했습니다.
값은 이 검사의 저장 시점까지 유한했으나 **장기 수렴이 확인되었다는 뜻이 아닙니다.**
Crop32 검사 결과를 실제crop768 학습의 성능·발산 판정으로 직접 사용하지 않습니다.
H2003step smoke에서도 LG/ALG guidance 상승이 관측된 만큼 본 실행의 loss와 상태를 확인해야 합니다.
코드는 비유한 loss/gradient/state를 감지하면 실패를 기록하며 beta/LR/clipping을 바꾸지 않습니다.

H200 crop768/batch8/FP32의 연산·역전파·freeze·마지막update 재개는
사용자 제공 `bapedragon_771` smoke에서4/4 통과했습니다.
이번 본학습 wrapper의216epoch H200 실행, full val500 평가55회,
수일 뒤 재개 및 실제 종료 시간은 아직 실행 결과가 없습니다.
