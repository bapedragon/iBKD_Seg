# Cityscapes SegFormer-B0 · 25-batch β 후보 생성

2026-09-23. H200 smoke v2 통과 뒤 실행할 **초기 손실 측정**입니다.
LG·ALG·iBKD λ=0.25·0.5의 β 후보를 4개씩 계산합니다.
학습 update·backward·validation·β 선택·2k 학습은 수행하지 않습니다.

## 이슈 입력값

[H200 요청 양식 열기](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)

| 항목 | 입력값 |
|---|---|
| 제목 | `[Request]: Cityscapes SegFormer-B0 25-batch beta calibration v1` |
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량 | **7 — H200 1장 전체** |

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_beta_calibration.sh
```

이 문서는 사용자가 제출할 입력안입니다. 외부 이슈를 자동 생성하지 않습니다.

## 측정·계산 규칙

- 실제 batch16·crop512×512·FP32·TF32 off. Train의 같은 증강 입력 25 batch,
  총 400개 sample presentation을 공유합니다. 중복 이미지가 가능하며 고유 이미지 400장이란 뜻은 아닙니다.
- 기존 확장 목록의 seed1 permutation 첫 400개 위치를 사용합니다. 입력 생성은 workers0입니다.
- Student seed1, adapter seed100001, 측정 RNG seed200001을 고정합니다.
  Student와 guide는 train mode, CIRKD teacher는 frozen eval입니다.
- LG/ALG의 locality는 동일하므로 한 경로에서 측정합니다. iBKD도 alignment·fusion을
  한 경로에서 측정해 두 λ의 guidance를 계산합니다. **25 batch × 2경로**로 네 조건을 구성합니다.
- iBKD는 각 batch에서 `(1−λ)×alignment + λ×fusion`을 계산한 뒤 중앙값을 구합니다.
  Alignment·fusion 각각의 중앙값을 혼합하는 계산과 구분합니다.
- `C=median(CE)`, `G=median(raw guidance)`, `r=[0.03,0.07,0.15,0.30]`에 대해
  `β=r×C/G`로 계산하고 **유효숫자 6자리**로 반올림합니다. 반올림 전 값도 보존합니다.
  실제 batch별 `β×G/CE`의 최솟값·중앙값·최댓값을 기록합니다.
- 유한한 양의 손실만 허용하며 `median(G)≤1e-12`, 중복 후보, 초기 상태·입력 불일치면
  후보를 확정하지 않습니다. 두 경로의 batch별 CE가 정확히 일치하는지도 검사합니다.
- `torch.no_grad()`로 측정합니다. Trainable parameter가 그대로이고 gradient가 없는지 확인합니다.
  측정 중 바뀐 student·guide BN buffer와 RNG는 **bitwise 복원·검사**합니다.
  실패 시에도 복원을 시도하고 마지막 정상 손실 및 실패 단계를 보존합니다.
- Gradient 전달 검사는 이미 통과한 [H200 smoke v2](reports/h200_smoke_v2/RESULTS.md)의 결과를 사용합니다.
  이번 측정에 backward를 추가하지 않습니다. Controller 관측도 0회입니다.
- FSKD·C2VKD·Vanilla는 이 β 후보 생성 대상에 포함하지 않습니다.

설정은 [calibration v1 JSON](configs/b0_beta_calibration_v1.json)에 사전 고정했습니다.
기존 공통·방법별 프로토콜 JSON은 변경하지 않았습니다. 후보 생성과 최종 β 선택은 별개입니다.

## 자료·실행 환경

Teacher는 CIRKD DeepLabV3-R101, student encoder는 smoke v2와 같은 NVIDIA 공식
HF ImageNet-only MiT-B0 역변환입니다. 모든 파일의 bytes/SHA-256과 encoder 전체 key를 검사합니다.
CIRKD Baidu의 B0 파일과의 동일성은 미확인입니다. 본실험은 이 초기화 출처를 명시한
revision으로 연결하거나 원본 동일성을 추가 확인해야 합니다.

기존 `/app/scratch/cityscapes_b0_smoke_v1/`의 준비 데이터·가중치를 재사용합니다.
데이터 provenance와 **실제 train/val 파일 전체 해시**를 확인합니다. Val 파일 확인은
입력 무결성 검사이며 모델 평가에 사용하지 않습니다. 준비 데이터가 없으면
`/app/data/chaoyang` 아래의 기존 ZIP을 찾아 감사·train/val 준비를 수행합니다.
CLIP 가중치와 FSKD의 torchsort CUDA 확장은 이 실행에 필요하지 않습니다.

Smoke v2의 strict deterministic·math SDPA·iBKD `flatmax_cpu_deform_v1` 경로를 유지합니다.
설치·자료 준비를 제외한 손실 측정은 앞선 smoke 속도에 근거해 수분 수준으로 예상하지만,
25-batch 실제 실행 전의 추정입니다. 새 컨테이너의 설치·전체 파일 검증·압축 해제에는
추가 시간이 필요하며 2–5분 내 전체 완료를 보장하지 않습니다.

## 결과와 성공 판정

기본 결과 폴더는 `/app/output/cityscapes_b0_beta_calibration_v1/`입니다.

| 파일 | 내용 |
|---|---|
| `run.log` | 설치·입력 생성·batch별 손실·최종 전체 JSON |
| `calibration_summary.json` | 전체 성공/실패, 단계, 실제 측정값, 검사와 소요 시간 |
| `beta_candidates.json` | 모두 통과한 경우에만 생성하는 네 조건의 β 후보·계산 근거·출처 해시 |
| `lg_alg/summary.json`, `ibkd/summary.json` | 경로별 25개 손실·batch 시간·상태 복원 검사 |
| `inputs/identity.json` | 공통 400개 입력의 sampler 위치·이미지 이름·tensor/파일 해시 |
| `inputs/batch_*.pt` | 공유한 실제 증강 입력, 총 약 2 GiB |
| `pip_freeze.txt` | 실제 실행 환경 |

마지막 로그 JSON에서 `status="passed"`, `beta_candidates_frozen=true`, 네 방법 각각
후보 4개, `optimizer_updates=0`, `backward_calls=0`, `selection_performed=false`를 확인합니다.
평가를 하지 않았으므로 `metrics=null`, `selected_epoch=null`이 정상입니다.
실패 시 원래 출력에 덮어쓰지 않습니다. 재실행은 예를 들어
`B0_CALIBRATION_OUTPUT=/app/output/cityscapes_b0_beta_calibration_v1_retry1`을 명령 앞에 지정합니다.

로컬 검사는 calibration 및 기존 B0 단위 검사 31개 통과, CUDA 전용 1개 skip입니다.
실제 가중치와 합성 batch2·crop64 입력으로 CPU 25-batch 두 경로도 통과했습니다.
모든 batch의 경로 간 CE 동일성, 가중치 유지, BN/RNG 복원과 네 조건 후보 계산을 확인했습니다.
H200의 실제 Cityscapes 25-batch 결과는 아직 생성하지 않았습니다.
