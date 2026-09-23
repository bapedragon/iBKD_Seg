# FSKD·C2VKD 공식 공개 자료 점검

확인일: 2026-09-23. 범위: 논문에 비교 방법으로 넣기 위한 공개 학습 자료의 완전성.
이번 작업은 소스·문헌 점검이며 학습 실행 검증은 아닙니다.

후속 사용자 요청으로 [방법별 재구현 프로토콜](BASELINE_METHOD_PROTOCOLS.md)을 작성했습니다.
FSKD는 공개 PiT 조합과 CLS 없는 B0의 attention 보완을 명세했고, C2VKD는 공개 loss·B0 head와
논문 기반 PDD 및 CLIP pool 대체안을 명세했습니다. **아래의 공개 자료 누락 판정이 해소된 것은
아닙니다.** 특히 C2VKD 대체 pool은 저자의 원본 가중치로 확인된 것이 아닙니다.

## 결론

**두 방법 모두 저자 저장소는 있지만, Cityscapes / SegFormer-B0를 그대로 재현할
수 있는 전체 코드·설정·가중치가 공개됐다고 판단할 수 없습니다.**
FSKD는 분류 구현 중심이고, C2VKD는 segmentation 구현 일부가 있으나 핵심 의존 코드가
누락됐습니다. 공통 학습 조건을 정하는 것만으로 이 방법별 공백이 해결되지는 않습니다.

| 항목 | FSKD | C2VKD |
|---|---|---|
| 저자 저장소 | [TouchNow/FSKD](https://github.com/TouchNow/FSKD) | [zhengxuJosh/C2VKD](https://github.com/zhengxuJosh/C2VKD) |
| 확인 commit | `969dddf278b2c9f2dadde504326fa9d704c5a5aa` | `fe1ab3d6f815969058451c221229a744fe872a47` |
| 공개 branch / commit 수 | master / 3개 | main / 6개 |
| 논문의 Cityscapes 주요 조건 | 확인한 프리프린트에 있음 | 확인한 arXiv v1에 있음 |
| 공개 코드 범위 | 분류 학습·구조 손실·CIFAR/ImageNet config | segmentation 모델·데이터 로더·학습 코드 일부 |
| Cityscapes-B0 전용 실행 config·명령 | 확인하지 못함 | 확인하지 못함 |
| 주요 미확인/누락 | segmentation feature/attention 연결과 loss 설정 | PDD 구현, attention 의존 코드·가중치, 전용 실행 경로 |
| 현재 판정 | 추가 저자 자료 또는 명시적 재구현 필요 | 추가 저자 자료 또는 명시적 재구현 필요 |

GitHub API에서 양쪽의 별도 tag·release·공개 issue/PR 목록은 모두 비어 있었습니다.
전체 공개 commit 이력도 확인했습니다. 이 판정은 확인한 공식 저장소의 공개 상태에
한정하며, 저자의 비공개 구현이나 접근하지 못한 저널 부록이 없다는 뜻은 아닙니다.

## FSKD 확인 결과

[공식 README](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/README.md)는
CIFAR-100 / PiT-Ti 및 ImageNet / DeiT-Ti 실행 예시를 제공합니다.
config는 `configs/cifar/`와 `configs/imagenet/`의 YAML 5개이며,
3개 commit의 Python·YAML·Markdown·shell 파일에서
Cityscapes/SegFormer/DeepLab/MMSegmentation 구현이나 설정을 찾지 못했습니다.
최신 commit은 기존 EXPERIMENT_PLAN에서 확인한 것과 같습니다.

분류 손실의 기본값과 실행 예시는 실제로 공개돼 있습니다.

| 범위 | global | patch | attention | stage |
|---|---:|---:|---:|---|
| `train.py` 기본값 | 1 | 1 | 1 | 1, 2, 3, 4 |
| README CIFAR-100 예시 | 1 | 1 | 40,000 | 3, 4 |
| README ImageNet 예시 | 100 | 1 | 1,000,000 | 1, 2 |
| Cityscapes / MiT-B0 저자 설정 | 미확인 | 미확인 | 미확인 | 미확인 |

출처: [train.py L104–124](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/train.py#L104-L124).
KD 가중치·temperature·GT 가중치·soft-rank regularization의 기본값도 각각 1입니다.
이 값들을 segmentation 저자 설정으로 간주하지 않습니다.

특히 [attention loss L90–114](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/distillers/simi.py#L90-L114)는
`student_attn[:, :, 0, extra_token:]`로 첫 토큰 attention을 사용합니다.
CLS token이 없는 MiT-B0에서 어떤 공간 attention 집계를 사용했는지 별도 근거가 필요합니다.
분류용 코드를 단순 연결하면 논문의 segmentation 구현과 같다고 보장할 수 없습니다.

남은 항목은 segmentation 전용 stage pairing·정렬, attention 정의, 각 항의 reduction과
가중치, KD temperature/CE-KD 계수 대응, 정확한 teacher checkpoint와 student 초기화입니다.

## C2VKD 확인 결과

[저자 프로젝트 페이지](https://vlislab22.github.io/C2VKD/)의 Code 링크가 위 저장소로
연결됩니다. README에는 논문 소개가 있지만 설치·실행·가중치 다운로드 안내는 없습니다.
공개 commit은 모두 2022-12-20에 작성됐습니다. 최신 GitHub `updated_at` 값은
소스 업데이트 날짜로 해석하지 않았습니다.

존재하는 파일은 `dataset/city_new/City_dataset.py`, Cityscapes split 목록,
`models/seg/MixT.py`, `models/seg/segformer.py`, DeepLabV3+·PVT/PVTv2 모델,
`train.py` 등입니다. 그러나 다음 공백을 실제 소스에서 확인했습니다.

| 문제 | 근거와 의미 |
|---|---|
| PDD의 실제 구현 누락 | [train.py L20–26](https://github.com/zhengxuJosh/C2VKD/blob/fe1ab3d6f815969058451c221229a744fe872a47/train.py#L20-L26)가 `utils.kd_losses.get_dkd_loss`를 import하지만 저장소에 `utils/`가 없음. `losses`, `ramps`, `mIOU_metrics`도 누락 |
| Attention 의존 코드 누락 | [att.py L9–10](https://github.com/zhengxuJosh/C2VKD/blob/fe1ab3d6f815969058451c221229a744fe872a47/models/deeplabv3plus/att.py#L9-L10)의 `models.attention.models.AttentionPool2d`, `models.deeplabv3plus.asppv3p` 파일이 없음 |
| 사용 가중치의 출처·준비 절차 미확인 | [train.py L156–169](https://github.com/zhengxuJosh/C2VKD/blob/fe1ab3d6f815969058451c221229a744fe872a47/train.py#L156-L169)는 로컬 teacher 파일과 `/pretrain_model0.pth`를 적재. 저장소·release·README에서 정확한 자산의 출처·배포/생성 안내를 찾지 못함. 로컬 경로만으로 저자가 직접 학습했다거나 외부 공개 가중치를 사용하지 않았다고 판단할 수 없음 |
| Cityscapes-B0 실행 경로 미제공 | `train.py`는 VOC 데이터, PVTv2 FPN, 21 classes를 사용. SegFormer 모델 파일이 있다고 전용 학습 경로까지 완성된 것은 아님 |

전체 6개 commit의 파일 이력에도 위 누락 모듈이 없었습니다.
이는 정적 점검 결과이며, GPU에서 실제 import 실패나 학습 실패를 실행해 측정한 것은 아닙니다.

[공개 train.py](https://github.com/zhengxuJosh/C2VKD/blob/fe1ab3d6f815969058451c221229a744fe872a47/train.py#L93-L99)의
기본값은 PDD 관련 alpha=0.5, beta=0.5, temperature=1, global/graph 가중치=0.1입니다.
linguistic 항에는 [코드 L208](https://github.com/zhengxuJosh/C2VKD/blob/fe1ab3d6f815969058451c221229a744fe872a47/train.py#L208)의
0.5가 직접 곱해집니다. 이것도 Cityscapes-B0 전용 확정값으로 볼 수 없습니다.

또한 [최종 목적함수 L238–248](https://github.com/zhengxuJosh/C2VKD/blob/fe1ab3d6f815969058451c221229a744fe872a47/train.py#L238-L248)는
CE를 계산·기록하지만 직접 더하지 않고 `dkd + gb_loss + loss_graph + language`를 사용합니다.
PDD 함수 본체가 누락돼 있어 내부 label 처리까지 검증할 수 없습니다.
C2VKD를 추가한다면 우리 공통 CE에 증류항을 단순히 더하기 전에 PDD 목적함수와의
관계를 별도 명세로 정해야 합니다.

## 논문에서 확인한 주요 학습 조건

아래 FSKD는 **2025 SSRN preprint**, C2VKD는 **2023 arXiv v1** 기준입니다.
저널 최종판의 전체 내용을 재검증했다는 의미가 아닙니다.

| 항목 | FSKD 프리프린트 | C2VKD arXiv v1 |
|---|---|---|
| Teacher | DeepLabV3 / ResNet101 | DeepLabV3+ / ResNet101 |
| Student | SegFormer MiT-B0·B1 | SegFormer MiT-B0·B1 등 |
| Cityscapes crop | 512×512 | 512×512 |
| Batch | 16 | 24 |
| 학습 step | 80,000 | 50,000 |
| 초기 LR / poly power | 6e-5 / 0.9 | 6e-5 / 1.0 |
| Optimizer | segmentation 문단에 종류를 별도 명시하지 않음 | AdamW |
| 초기화 | Cityscapes encoder 초기화 별도 확정 불가 | encoder ImageNet-1K / decoder random |
| 평가 | mIoU, Figure 7에서 400 step 간격 | val mIoU, 512×512 sliding window |

FSKD 출처: [SSRN 5385770](https://ssrn.com/abstract=5385770), PDF p.9–10 §4.1,
p.16 Figure 7. 기존 명세와 같은 로컬 파일임을 확인했습니다:
2,867,689 bytes, SHA-256 `02667b51c9f1402dd29c01d07a1badd623a39ce0d6f50eafb80d1f8ae6852844`.
C2VKD 출처: [arXiv v1 §IV-B/C](https://arxiv.org/html/2310.07265v1#S4).

[FSKD 저널 페이지](https://www.sciencedirect.com/science/article/pii/S0893608026000638)와
[C2VKD 저널 페이지](https://www.sciencedirect.com/science/article/pii/S0031320324007805)는
직접 본문 접근이 제한됐습니다. 별도 공개 부록에서 누락 설정을 확인하지 못했으며,
최종판에도 그 정보가 없다고 단정하지 않습니다.

## Teacher 체크포인트 출처 추가 대조

2026-09-23 후속 질문에 따라 **teacher를 직접 학습했는지**와
**이미 학습된 checkpoint를 어디서 가져왔는지**를 구분해 재확인했습니다.
공개 코드의 `torch.load()`와 로컬 파일명은 사전 학습 가중치 적재의 근거지만,
그 가중치를 저자가 학습했는지 다운로드했는지의 근거는 아닙니다.

| 자료 | Teacher | Cityscapes val mIoU | VOC val mIoU |
|---|---|---:|---:|
| CIRKD 2022 공개 README | DeepLabV3-ResNet101 | 78.07 | 77.67 |
| FSKD 2025 SSRN Table 5 | DeepLabV3-Res101 | 78.1 | 77.7 |
| C2VKD 2023 arXiv Tables I·II | DeepLabV3+-ResNet101 | 77.48 | 75.77 |

출처: [CIRKD 고정 버전 README](https://github.com/winycg/CIRKD/blob/48eb81b7a0ed7c59f9e347e97feb95a23e943d12/README.md),
FSKD 로컬 PDF p.10 §4.1 및 p.12 Table 5,
[C2VKD arXiv v1](https://arxiv.org/html/2310.07265v1#S4).
CIRKD README는 Cityscapes teacher의
[공개 checkpoint 링크](https://drive.google.com/file/d/1zUdhYPYCDCclWU3Wo7GbbTlM8ibQ_UC1/view?usp=sharing)도 제공합니다.
현재 main은 CIRKDV2 내용으로 바뀌어 있으므로 위 비교에는 2022 commit을 사용했습니다.

- **FSKD:** CIRKD 실험 설정을 따른다는 문구와 두 데이터셋 teacher 수치의 반올림 일치는
  CIRKD 공개 teacher 재사용이라는 해석을 뒷받침합니다. 다만 확인한 자료에는 정확한
  checkpoint URL·파일 식별자·명시적 재사용 문장을 찾지 못했으므로, 동일 파일 사용은
  확정 사실과 구분한 추정입니다. 직접 새로 학습했다고 볼 근거도 확보하지 못했습니다.
- **C2VKD:** 확인한 arXiv v1의 teacher 구조·성능은 CIRKD와 다릅니다.
  공개 모델 코드에도 저해상도 feature와 저층 feature를 결합하는 V3+ decoder가 있어
  이름의 `+` 차이만 있는 것은 아닙니다. CIRKD의 완성된 V3 teacher checkpoint를
  그대로 사용했다고 확인할 수 없습니다. 이것이 backbone 가중치의 일부 재사용 가능성까지
  배제하는 것은 아니며, 직접 학습/외부 checkpoint 사용 중 어느 쪽인지는 미확인입니다.
- C2VKD의 공개 학습 파일은 VOC 경로이므로 그 파일의 로컬 경로를 Cityscapes teacher
  출처의 직접 증거로 확장하지 않습니다. 저널 최종판·추가 저자 자료에 관한 한계도 유지합니다.

우리 실험은 이미 CIRKD 공개 DeepLabV3-R101을 공통 teacher로 선택했습니다.
**두 논문이 정확히 같은 파일을 사용했는지 미확인이라는 이유만으로 우리 teacher를
새로 학습해야 한다는 뜻은 아닙니다.** C2VKD를 그 teacher에 적용한다면 원 논문 조건과
구분한 공통 조건 비교로 기록합니다.

## 우리 비교표에 적용할 판단

- 두 방법 모두 CNN→ViT segmentation 비교 후보로서 관련성이 있습니다.
- 논문 보고값을 인용하는 표에서는 원 논문의 teacher·학습·평가 조건과 출처 버전을
  표시합니다. 이를 우리 공통 프로토콜에서 재현한 값으로 표시하지 않습니다.
- 우리 공통 조건으로 다시 실행할 경우, 추가 저자 자료를 확보하거나 미확인 부분의
  재구현 선택을 고정한 후 학습합니다. 직접 보완했다면 표·본문에 재구현임을 명시합니다.
- C2VKD는 teacher 종류, batch, step, poly power, 평가 방식부터 기존 공통 조건과
  다릅니다. 원 논문 수치와 우리 결과를 같은 조건의 성능 순위로 읽게 해서는 안 됩니다.
- FSKD에는 Cityscapes script/config·loss/attention 연결·checkpoint를,
  C2VKD에는 누락 `utils/`·attention 모듈·attention/teacher 가중치 준비 절차·Cityscapes-B0
  실행 설정을 우선 추가 확보해야 합니다.

이번 점검은 기존 공통 프로토콜과 여섯 비교 조건을 변경하지 않습니다.
C2VKD는 공개 자료 점검 대상이며 아직 comparison manifest에 추가하지 않았습니다.
B0 구현·GPU 실험은 시작하지 않았습니다.
