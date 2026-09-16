# H200 이슈 입력안 — 공식 소스·공개 가중치 기반 L/16 smoke v2

이슈는 사용자가 직접 제출합니다. 이 문서와 실행 스크립트는 이슈를 자동 등록하지 않습니다.

[공식 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)에 아래 내용을 입력합니다.

| 항목 | 입력값 |
|---|---|
| 제목 | `[Request]: Cityscapes 공식 소스 DeepLabV3 → Segmenter-L/16 LG·ALG·iBKD smoke v2` |
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량(MIG 갯수) | `7` — H200 한 장 전체 |

코드 실행 명령어:

```bash
bash phase4/phase4_cityscapes/scripts/run_cityscapes_official_l16_smoke.sh
```

## 이전 smoke와 바뀐 점

이전 `real_smoke_v1`은 S/16을 사용한 것 외에도 torchvision teacher, 이식한
Segmenter, 무작위 가중치, 축소된 증강·학습 설정을 사용했습니다.
따라서 당시 통과는 데이터와 학습 연산의 연결 확인이며 공개 결과 재현이 아닙니다.

이번 실행은 저자의 원본 Segmenter `VisionTransformer`·`MaskTransformer`·`Segmenter`,
Cityscapes 데이터 클래스와 MMSeg 증강, timm 0.4.12 optimizer, 원본 polynomial scheduler,
원본 sliding-window inference를 직접 호출합니다. 원본 저장소 파일은 수정하지 않습니다.

- Teacher: MMSeg DeepLabV3 **ResNetV1c-101-D8**, 공개 Cityscapes 80k 학습 checkpoint.
  ASPP와 auxiliary head를 포함한 전체 state를 `strict=True`로 로드합니다.
- Student: 저자 소스 **ViT-L/16, 24블록, 1024채널 + 2블록 mask decoder**.
  저자가 지정한 Google AugReg ImageNet 사전학습 NPZ를 원본 timm 로더로 로드합니다.
  Cityscapes를 이미 학습한 student checkpoint로 시작하지 않습니다.
- Crop768, 실제 batch8, FP32, SGD **Nesterov** lr0.01/momentum0.9/wd0,
  min_lr1e-5/poly power0.9. Gradient clipping 없음.
- 스케줄 분모는 원본 설정의 216epoch × ceil(2975/8)=80,352step을 유지합니다.
  **실행은 각 방법 3step에서 종료**합니다. 3step에 맞춰 학습률을 압축하지 않습니다.
- 원본 증강: resize ratio0.5–2.0, crop768/category ratio0.75, flip,
  photometric distortion, ViT normalization, pad. Teacher는 동일한 증강 이미지에
  teacher의 ImageNet normalization을 적용합니다.
- LG/ALG는 블록 `[0, 12, 23]`; iBKD는 24개 블록 전체의 학습 가능한 가중합.
  기존 first/middle/last 및 all-block 규칙을 L/16 깊이에 확장한 연구 코드입니다.
  LG·ALG·iBKD의 Cityscapes 설정 자체가 저자 공식 프로토콜이라는 뜻은 아닙니다.

## 이번 실행에서 검증하는 범위

1. `/app/data/chaoyang`의 ZIP 2개를 byte size·SHA-256·CRC로 확인합니다.
2. `/app/scratch/cityscapes_official_l16_v2/cityscapes`에 train/val을 준비하고
   원본 이미지·labelIds 2,975/500쌍을 검사합니다. 선택 표본의 labelTrainIds도 변환합니다.
3. 고정 commit의 원본 저장소 3개와 공개 가중치 2개(약 1.57GB)를 자동 준비합니다.
   모델 다운로드 실패나 해시 불일치 시 중단합니다. 무작위 가중치로 대체하지 않습니다.
4. Vanilla/LG/ALG/iBKD를 별도 프로세스에서 순차 실행합니다.
   동일한 첫 train 24장, 동일한 증강·student 초기 state를 방법 간 대조합니다.
5. 각 방법 3step의 loss·gradient·teacher freeze·메모리 사용량을 검사합니다.
   마지막 step은 checkpoint에서 model/optimizer/scheduler/RNG/controller를 복원해 재실행합니다.
6. 동일한 첫 val 2장을 원본 해상도에서 window768/stride512로 평가합니다.
   Pixel accuracy를 우선 기록하고 같은 예측의 mIoU도 기록합니다.
   이번 부분 평가 점수로 방법 순위나 checkpoint를 선택하지 않습니다.

원본 계산을 유지하면서 메모리를 줄이기 위해 transformer block activation recomputation을
사용합니다. iBKD는 모든 key를 유지하는 query chunking을 사용합니다.
현행 PyTorch에서 삭제된 import/SyncBN 초기화 API는 별도 호환 계층으로 연결합니다.
Teacher SyncBN은 eval 상태이며 학습되지 않습니다. 원본 forward 계산은 유지합니다.

## 공개 논문 결과와의 구분

기준은 [저자의 공개 코드](https://github.com/rstrudel/segmenter)의 `config.yml` 및
`train.py` 기본값입니다. **배포된 Cityscapes 결과의 완전 재현이라고 부르지 않습니다.**

저자 model-zoo `variant.yml` 링크는 2026-09-16 확인 시 HTTP403을 반환했습니다.
공개 코드의 위 구성은 334,845,966 parameters이며 README의 Cityscapes 표는 322M을
기재하고 있어, 배포 결과의 세부 override까지 동일하다고 확인할 수 없습니다.
이 차이를 임의로 decoder 크기 변경으로 맞추지 않습니다. 본실험·논문 수치 직접 비교 전에
배포 설정을 확보하거나, 공개 코드 기본값 재현을 비교 기준으로 명시해야 합니다.

학습을 3step으로 제한하고 val 2장만 평가하므로 성능 재현·수렴 검증이 아닙니다.
ALG/iBKD 장기 종료 시점은 관측하지 않으며 별도 controller 진단으로만 검사합니다.
Test split은 모델 입력·평가에 사용하지 않습니다.

## 로그와 결과

완료 표식:

```text
[CITYSCAPES_OFFICIAL_L16_SMOKE_DONE] status=passed methods=4/4 scientific_result=false
```

출력 루트: `/app/output/cityscapes_official_l16_smoke_v2`

- `run.log`, `upload_check.json`, `manifest.json`, `preparation.json`
- `artifacts/smoke_summary.json`: 네 방법의 검사 결과
- `artifacts/<method>/summary.json`: loss, 진단 정확도/mIoU, 메모리, 재개 확인
- `artifacts/<method>/provenance.json`: 공식 코드 commit·파일 해시 및 공개 가중치 해시
- `artifacts/<method>/data_identity.json`: 표본·증강 tensor·변환 라벨 해시
- `artifacts/<method>/upstream_recipe.json`: 원본에서 읽은 모델·데이터 설정

재개 검사용 대형 checkpoint는 검사를 통과한 뒤 제거하며 byte size/SHA-256과 검사 결과만
남깁니다. 검사 실패 시에는 해당 checkpoint를 유지합니다. 저장소에 데이터·가중치를 넣지 않습니다.
OOM 발생 시 batch/crop/정밀도를 자동 변경하지 않고 실패를 보고합니다.

같은 컨테이너에서 재실행하려면 `CITYSCAPES_OFFICIAL_OUTPUT`에 새 출력 폴더를 지정합니다.
공유 ZIP은 읽기만 하며 공식 소스·가중치·압축 해제 데이터는 `/app/scratch`에 준비합니다.
