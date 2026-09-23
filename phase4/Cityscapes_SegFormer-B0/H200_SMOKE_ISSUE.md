# Cityscapes SegFormer-B0 · H200 smoke v2 요청

2026-09-23. **7개 경로의 연결·손실·gradient·재개·평가를 확인하는 실행**입니다.
β 선별이나 80k 본실험을 시작하지 않습니다. v1에서는 실제 H200에서 7개 방법 모두
3 update를 완료했지만 재실행 가중치 비교에 실패했습니다.
[요청 819 점검 결과](reports/h200_smoke_v1_819/RESULTS.md)를 반영한 v2 재검사입니다.
v2 로컬 CPU 연결·재실행 7/7과 단위 검사 14개가 통과했으며 GPU 결과는 아직 없습니다.

## 이슈 입력값

[H200 요청 양식 열기](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)

| 항목 | 입력값 |
|---|---|
| 제목 | `[Request]: Cityscapes SegFormer-B0 7-method smoke v2` |
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량 | **7 — H200 1장 전체** |

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_smoke.sh
```

이 문서는 사용자가 제출할 입력안이며, 외부 이슈를 자동 생성한 것은 아닙니다.

## 실행 내용

| 경로 | 확인 대상 |
|---|---|
| Vanilla | B0 encoder·decoder와 CE |
| LG | block 1·5·8의 projection·locality loss |
| ALG | LG 연결과 186-step controller 상태 |
| iBKD λ=0.25 | 전체 8개 block → 256채널·16×16 → aggregation·alignment·fusion |
| iBKD λ=0.5 | 같은 연결에서 다른 λ의 gradient·재개 |
| FSKD* | stage 3·4의 global·patch·attention·logit KD |
| C2VKD* (CLIP-pool) | PDD·global·patch·linguistic와 frozen pool, **추가 후보** |

각 방법을 별도 프로세스로 순차 실행합니다. 방법 하나가 실패해도 나머지를 계속 검사합니다.
공통 설치·자료 검증이 실패하면 학습을 시작하지 않고 실패 단계를 기록합니다.

- 실제 **batch16, crop512×512, FP32, TF32 off**, gradient accumulation 없음.
- AdamW LR6e-5·WD1e-4, 80k poly schedule의 처음 **3 update**만 수행.
- LG·ALG·iBKD는 같은 3개 batch로 업데이트 없이 손실 크기를 측정하고,
  `β = 0.07 × median(CE) / median(raw guidance)`를 smoke용으로만 사용.
  측정 뒤 student의 BN 등 상태와 학습 RNG를 복원합니다. 본실험 β 후보가 아닙니다.
- 모든 방법이 같은 초기 student·48개 입력·crop·flip을 사용했는지 해시 비교.
- 모든 손실 항이 student encoder에 유한한 0이 아닌 gradient를 전달하는지 확인.
  encoder·decoder·guidance parameter group의 gradient와 실제 update도 확인.
- frozen CIRKD teacher·BN 및 C2VKD pool이 변하지 않는지 검사.
- strict deterministic algorithms와 cuBLAS 재현성 설정, math SDPA 사용.
  CE는 같은 유효 픽셀 평균을 2D logits로 계산합니다.
- iBKD는 기존 L/16의 결정적 CBAM 경로를 사용합니다. flat max와 작은 spatial deformable
  convolution의 CPU 계산이며 GPU↔CPU autograd를 유지합니다. 수식·계수는 유지합니다.
- 2번째 update 뒤 모델·adapter·optimizer·RNG·controller·입력 위치 저장.
  재실행 직전의 모델·guide·optimizer·controller·RNG는 저장값과 bitwise 비교합니다.
  3번째 update를 재개 후 재실행해 loss와 전체 상태 비교(`rtol=2e-5, atol=2e-6`).
  수치 비교 실패도 파라미터 이름·손실 이력과 함께 남깁니다. 그 경우 원래 연속 학습의
  3번째 상태로 val2를 진단하고 전체 판정은 실패로 유지합니다.
- ALG·iBKD의 186·372-step 관측 및 373-step off 경계는 **별도 합성 loss 검사**로 확인.
  3-update 학습에 관측 주기를 축소 적용하지 않습니다. 두 방법 warm-up은 0입니다.
- val 첫 2장을 원본 1024×2048에서 좌우 1024 crop으로 평가.
  저해상도 logits를 먼저 이어 붙인 후 한 번 확대하고, ignore=-1·19-class mIoU를 확인.
  전체 val500 평가·best checkpoint 선택·test 평가를 수행하지 않습니다.
- FSKD의 `torchsort==0.1.10`을 설치하고 실제 CUDA forward/backward를 검사합니다.
  CUDA 확장 미설치 등의 환경 문제도 해당 방법의 실패로 기록합니다.

FSKD*는 공개값과 논문으로 보완한 재구현입니다. C2VKD*는 원본 pooling 가중치를
CLIP RN101 pool로 대체한 추가 후보이며, 주 비교 6개와 구분합니다.
C2VKD의 CE는 진단용으로만 기록하고, 최적화에는 명세대로 PDD를 사용합니다.

## 데이터·가중치와 이번 smoke의 예외

기존 Cityscapes ZIP 두 개를 `/app/data/chaoyang` 아래에서 찾습니다.
기존 감사의 bytes/SHA-256·CRC·split 수를 검사하고 train/val만 추출합니다.
test 파일을 학습·평가에 사용하지 않습니다. 검증된 추출 cache는 재사용할 수 있습니다.

가중치는 [고정된 출처·bytes·SHA-256](configs/b0_asset_sources_v1.json)으로 다운로드하며,
불일치하면 중단합니다. random 초기화나 부분 적재로 넘어가지 않습니다.

| 대상 | 실제 적재 파일 | 로컬 확인 |
|---|---|---|
| Teacher | CIRKD DeepLabV3-R101, 245,151,519 bytes | SHA-256·전체 key strict load 통과 |
| Student encoder | NVIDIA `mit-b0` ImageNet 배포본, 14,380,029 bytes | SHA-256·역변환·encoder 전체 key 검사 통과 |
| C2VKD pool | OpenAI CLIP RN101, 291,791,292 bytes | SHA-256·pool strict load 통과 |

**Student 다운로드 출처는 smoke의 명시적 예외입니다.** NVlabs README의 Drive 폴더가
404여서 NVIDIA 계정의 [ImageNet-only MiT-B0](https://huggingface.co/nvidia/mit-b0)를
고정 revision에서 받습니다. HF 변환의 key 변경과 K/V 분리를 역으로 적용하며,
ImageNet classifier를 제외한 encoder만 적재하고 segmentation decoder는 새로 초기화합니다.
CIRKD Baidu의 `mit_b0.pth`와 파일/텐서 동일성은 아직 검증하지 않았습니다.
본실험으로 이어갈 때에는 초기화 출처 확인 또는 protocol revision이 필요합니다.

공통 JSON은 변경하지 않았습니다. 이번 실행의 짧은 β 측정, val2, 입력 cache 생성의
workers0 등 차이는 [smoke v2 명세](configs/b0_smoke_v2.json)에 별도로 고정했습니다.
방법 수식은 유지하면서 iBKD fusion과 Gram loss를 정확한 chunk 계산으로 처리합니다.
student backbone의 gradient checkpointing은 사용하지 않습니다.

## 결과 확인

기본 출력: `/app/output/cityscapes_b0_smoke_v2/` (v1 결과 보존)

- `run.log`: 전체 설치·검증·학습 로그. **마지막 줄은 전체 결과 JSON**입니다.
- `smoke_summary.json`: 방법별 loss·진단 metric·gradient·재개 상태·실패 이유.
- `<method>/summary.json`, `<method>/resume_step2.pt`: 방법별 요약과 재개 검사 checkpoint.
- `input_identity.json`, `inputs.pt`, `zip_audit.json`, `pip_freeze.txt`: 재현성 자료.

판정은 `status=passed`, `passed_methods=7`, `primary_status=passed`,
`c2vkd_supplementary_status=passed`인지 확인합니다. 주 비교 6개가 통과하고 C2VKD만
실패하면 그 상태를 구분해 표시하며, 전체 성공으로 처리하지 않습니다.
`completed_training_methods`, `resume_passed_methods`, `evaluated_methods`로 학습·재개·평가의
진행 범위를 따로 확인합니다. 요약에는 선택 epoch가 `null`로 남습니다. val2 수치를 논문 성능이나 방법 순위로 사용하지 않습니다.

기본 cache는 `/app/scratch/cityscapes_b0_smoke_v1/`입니다. 필요하면
`CITYSCAPES_ZIP_DIR`, `B0_SMOKE_DATA`, `B0_SMOKE_CACHE`, `B0_SMOKE_OUTPUT`으로 경로를
지정할 수 있습니다. 재실행은 새로운 출력 경로를 사용하며 기존 결과를 덮어쓰지 않습니다.
전체 작업 제한은 9시간이고, 본실험으로 자동 전환되지 않습니다.

로컬 검증 범위와 재실행 방법은 [구현 검사 기록](SMOKE_PREPARATION.md)에 있습니다.
