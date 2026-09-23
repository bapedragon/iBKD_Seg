# FSKD·C2VKD 방법별 재구현 프로토콜 v1

작성: 2026-09-23. 사용자 요청에 따라 공개된 논문·코드에서 확인한 부분을 유지하고,
비공개 부분에는 **첫 구현에 사용할 구체적인 보완안**을 정했다.
이 문서는 구현 명세 초안이다. B0 구현·가중치 적재·GPU 실험을 완료했다는 뜻이 아니다.
공통 JSON의 내용과 SHA-256은 변경하지 않았다.

## 1. 적용 범위와 출처 구분

표의 `공개`는 실제 소스에 있는 값, `이식`은 다른 모델/데이터의 공개값을 가져온 선택,
`보완`은 이번에 정한 재구현 규칙이다. 이식·보완을 저자의 Cityscapes 설정으로 표시하지 않는다.

| 항목 | FSKD | C2VKD |
|---|---|---|
| 실행용 명세 ID | `fskd_b0_reimplementation_v1` | `c2vkd_b0_clip_pool_reconstruction_v1` |
| Teacher / student | 공통 CIRKD DeepLabV3-R101 / MiT-B0 | 동일 |
| 기본 데이터·학습·평가 | [공통 프로토콜](PROTOCOL.md) 그대로 | 동일하되 아래 목적함수 예외 명시 |
| 방법 손실 | CE + logit KD + global + patch + attention | PDD + global + patch + linguistic |
| 사용 feature | stage 3·4 마지막 출력 | stage 4 마지막 출력 |
| 추가 사전학습 가중치 | 없음 | 대체안에서 CLIP RN101의 attention pool만 사용 |
| 비교표 표기 | `FSKD*` | `C2VKD* (CLIP-pool)` |
| 현재 위치 | 기존 6개 비교의 FSKD 재구현 명세 | 별도 추가 후보; 기존 6개 주 비교에 자동 편입하지 않음 |

공통 crop 512, 실제 batch 16, 80k step, AdamW 6e-5, poly 0.9, warm-up 0,
400-step 전체 val, 동일 초기 student·데이터 순서·증강·CIRKD teacher를 유지한다.
두 방법에는 ALG/iBKD의 종료 controller나 iBKD 전용 256채널·16×16 adapter를 적용하지 않는다.
모든 KD 항은 step 1부터 80k까지 사용한다. 손실 계수는 아래 고정값으로 먼저 검증하며
LG/ALG/iBKD처럼 β 4개 탐색을 의무적으로 추가하지 않는다.

C2VKD 원문의 DeepLabV3+ teacher, batch 24, Cityscapes 50k, poly power 1.0 등은
이번 공통 비교 조건으로 대체된다. 따라서 결과는 원 논문 수치의 직접 재현이 아니라
공통 teacher/student 조건으로 이식한 결과로 보고한다.

확인한 원문은 FSKD의 [SSRN preprint](https://ssrn.com/abstract=5385770),
C2VKD의 [arXiv v1](https://arxiv.org/html/2310.07265v1)이다.
FSKD PDF는 2,867,689 bytes, SHA-256
`02667b51c9f1402dd29c01d07a1badd623a39ce0d6f50eafb80d1f8ae6852844`.
저널 최종판 전문·부록을 모두 확인한 것은 아니다.
공개 범위 점검은 [기존 감사 기록](OFFICIAL_BASELINE_AUDIT.md)을 참고한다.

## 2. FSKD: 첫 실행용 설정

### 2.1 연결과 계수

| 항목 | 선택값 | 근거 |
|---|---|---|
| Global / patch 대응 | MiT stage 3 → ResNet layer3, stage 4 → layer4 | **이식·보완**: README PiT 예시와 논문 Table 7의 뒤쪽 두 그룹 선택 |
| 학생 출력 위치 | 각 stage 마지막 block 뒤의 stage LayerNorm 출력 | **보완**: CIRKD MiT의 반환 feature 사용 |
| 공간 정렬 | flatten → Linear(Ns, Nt) → GELU → reshape | **공개**: `patch-align=lg` |
| 채널 정렬 | Conv3×3(Cs,Ct,padding=1,bias=True) → GELU | **공개**: `channel-align=cg` |
| λglobal / λpatch / λattention | **1 / 1 / 40,000** | **이식**: README CIFAR-100 / PiT-Ti의 한 조합 전체를 채택 |
| CE / logit KD 계수 | **1 / 1** | **공개 기본값**을 이식; 논문 Eq.1의 α를 추가로 곱하지 않음 |
| KD temperature | **1** | 공개 파서 기본값 이식 |
| soft-rank | `regularization='l2'`, strength **1** | 공개 호출과 torchsort 기본 regularization 이식 |

논문의 네 block 그룹은 B0의 개별 8개 block을 뜻하지 않는다. 여기서는 네 stage를
그룹으로 대응한다. 512 crop에서 학생 stage 3은 `[B,160,32,32]`, stage 4는
`[B,256,16,16]`이다. 고정한 CIRKD dilated ResNet 코드상 teacher layer3/4는
각각 `[B,1024,64,64]`, `[B,2048,64,64]`가 예상된다. 실제 적재 후 shape를 검사한다.
Teacher의 기본 반환값에 포함된 **ASPP 뒤 256채널 feature를 layer4로 오인하지 않는다.**

Global용 Linear는 각각 `1024→4096`, `256→4096`, Conv는 `160→1024`, `256→2048`이다.
모든 stage를 선택하면 초기 고해상도 feature의 dense 공간 Linear가 매우 커지므로,
공개 PiT 예시와 일관된 뒤쪽 두 stage 조합으로 시작한다. B0 최적 조합이라는 근거는 없다.
채널·공간 정렬의 순서와 GELU는 공개 구현대로 유지한다.

출처: [README](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/README.md),
[SimiKD·FeatureAlignment](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/distillers/simi.py),
[KD·초기화](https://github.com/TouchNow/FSKD/blob/969dddf278b2c9f2dadde504326fa9d704c5a5aa/distillers/utils.py).

### 2.2 손실의 축과 reduction

`MSEmean`은 모든 입력 원소의 평균, `B=16`, `V`는 ignore=-1을 제외한 픽셀 집합이다.

```text
L = CE_valid + KD_valid + G + P + 40000*A
KD_valid = T² × mean_{pixel in V} sum_class pT × (log pT − log pS), T=1
G = sum_{j in {3,4}} MSEmean(Align_j(S_j), T_j) / B
P = sum_{j in {3,4}} MSEmean(RS_j, RT_j)
```

Logit KD는 양쪽 logits를 label 격자 512×512에 bilinear/align_corners=True로 맞춘 뒤,
유효 픽셀을 `[|V|,19]`로 모아 분류 코드의 `batchmean`을 적용한다.
4D logits에 바로 `batchmean`을 적용해 픽셀 수만큼 손실을 키우지 않는다.
CE도 공통 valid-pixel mean이다. 유효 픽셀이 없는 batch는 데이터 오류로 중단한다.

`G`의 **MSE 뒤 추가 `/B`와 stage 합산은 공개 코드의 동작**을 보존한 것이다.
논문 Eq.4의 그룹 평균 표현과 다르지만, 계수와 reduction을 함께 코드 기준으로 이식한다.
멀티 GPU로 옮길 때도 이 분모는 local batch가 아닌 실제 global batch 16이다.

`P`는 teacher를 해당 학생의 원래 격자(32×32 또는 16×16)에 bilinear/False로 줄인 뒤,
양쪽의 각 위치 채널 벡터를 L2 정규화하고 **이미지별** cosine Gram matrix를 비교한다.
채널 projection 이전의 학생 feature를 쓴다. 각 stage loss는 B·N² 전체 평균이며 두 stage를 합한다.
L2 norm에는 보완값 `eps=1e-8`을 둔다. 관계 행렬은 정확한 chunk 계산이 가능하며
무작위 위치 선택이나 추가 pooling으로 정의를 바꾸지 않는다. Feature 손실에 GT ignore mask는 씌우지 않는다.

### 2.3 CLS 없는 MiT의 attention 보완

공개 FSKD의 `attn[:,:,0,...]`는 **CLS가 있는 분류 모델용**이다.
B0의 첫 공간 토큰을 CLS로 간주하지 않는다. Stage 4 마지막 block의 attention을 사용한다.
이 stage의 spatial-reduction ratio는 1이며, head 8개, query/key 각각 256개다.

```text
attention = softmax(Q Kᵀ / sqrt(d), key축)       # dropout 이전, [B,8,256,256]
s_importance[k] = mean_{head, query} attention[head,query,k]
t_importance = softmax_spatial(mean_channel(resize(T4,16×16)²))
rS = soft_rank(s_importance, l2, strength=1)
rT = soft_rank(t_importance, l2, strength=1)
A = mean_batch(6 × sum_position((rS-rT)²) / (256 × (256²−1)))
```

**Query와 head를 평균하여 각 key 위치가 받는 attention을 측정**하는 선택은 이번 보완이다.
Key 축을 평균하면 softmax의 합 제약으로 상수가 되므로 사용하지 않는다.
학생 attention의 gradient를 보존하고 teacher만 detach한다. 기존 student forward의
attention을 노출하며 다른 dropout/새 attention을 덧붙이지 않는다.
마지막 식은 코드의 `1 - mean(1 - ...)`와 대수적으로 같고 FP32 상쇄 오차를 줄인다.
Hard argsort로 바꾸어 학생 gradient를 끊지 않는다.

### 2.4 학습과 상태

Adapter는 학생과 같은 AdamW 그룹에 포함한다. LR/WD 배수, 별도 optimizer, adapter warm-up은 없다.
Conv weight는 공개 코드의 Kaiming-normal fan_out/ReLU, Linear weight는 trunc-normal std=.02,
Linear bias=0, Conv bias는 해당 PyTorch Conv 생성 기본값을 사용한다.
초기 공통 student state를 먼저 복원하고 adapter RNG를 별도로 고정한다.
두 방법 모두 adapter seed는 `100000 + run_seed`이며, 생성 후 학습용 RNG를 복원한다.
Teacher 전체는 eval/no_grad이며 parameter와 BN 통계가 바뀌면 오류다.
추론에는 student encoder/decoder만 사용한다.

**공식 Cityscapes 값이 확인되지 않은 사실은 그대로 남는다.** 위 수치는 미확인 칸을
숨긴 것이 아니라, 사용자가 요청한 첫 재구현값을 출처와 함께 정한 것이다.
[FSKD JSON](configs/fskd_method_protocol_v1.json)에 같은 설정을 기록했다.

## 3. C2VKD: 손실·연결 초안과 가중치 대체안

### 3.1 공개 코드와 논문이 다른 지점

- PDD 함수 `utils.kd_losses.get_dkd_loss`는 공개 tree에 없다. 함수 이름만 보고 일반 DKD를 넣지 않는다.
- Eq.3·4는 `KL(student || teacher)` 방향이지만 공개 train.py의 KL 호출은 반대 방향이다.
  **구현이 공개된 global/linguistic 항은 코드의 teacher→student 방향을 채택한다.**
- 공개 train.py는 PVT 계열의 두 반환값을 사용한다. SegFormer 파일의 세 반환값을 그대로 받는 runner가 아니다.
  B0 연결은 공개 SegFormer의 두 align head와 논문 Algorithm 1을 기준으로 작성한다.
- Eq.10 및 실제 `loss_vit`는 PDD와 VLFD 세 항을 합하며 CE를 별도 합산하지 않는다.
  **C2VKD에 한해 공통 CE 계수 1을 0으로 override하고 CE는 진단 로그만 남긴다.**
  이는 목적함수 예외이며 backbone·teacher·데이터·optimizer·평가 변경은 아니다.

출처: [논문 III-A/B·Algorithm 1](https://arxiv.org/html/2310.07265v1),
[train.py](https://github.com/zhengxuJosh/C2VKD/blob/fe1ab3d6f815969058451c221229a744fe872a47/train.py),
[SegFormer 연결](https://github.com/zhengxuJosh/C2VKD/blob/fe1ab3d6f815969058451c221229a744fe872a47/models/seg/segformer.py).
위 충돌 때문에 본 초안이 저자의 미공개 학습 경로와 동일하다고 주장할 수 없다.

### 3.2 Feature 연결과 고정값

| 항목 | 선택값 | 출처 / 보완 |
|---|---|---|
| 학생 visual | S4 `[B,256,16,16]` → Conv1×1(256,2048,bias=False) → 32×32 | 공개 SegFormer 코드 |
| Teacher visual | ResNet layer4 `[B,2048,64,64]` → 32×32 | 마지막 encoder feature라는 원문에 맞춘 CIRKD 연결 |
| 학생 linguistic | S4 → GAP → Conv1×1(256,512,bias=False) → `[B,512]` | 공개 SegFormer의 AvgPool16과 512 crop에서 동일 |
| Teacher linguistic | layer4 → 16×16 → attention pooling → `[B,512]` | 구조는 공개 att.py, 입력 resize는 보완 |
| PDD α / β / temperature | **0.5 / 0.5 / 1** | 공개 train.py 기본값; 논문 Table V의 1:1 비율과 일치 |
| λglobal / λpatch / λlinguistic | **0.1 / 0.1 / 0.5** | 공개 train.py 실제 계산값 |

Feature resize는 bilinear/align_corners=False. 2048채널 visual과 512차원 linguistic은
서로 다른 head다. iBKD의 256채널·16×16 규격으로 일괄 치환하지 않는다.
사용하지 않는 `alignhead1`/classifier는 optimizer에 넣지 않는다. 사용하는 두 head는
PyTorch Conv2d 기본 초기화, 학생과 동일 LR/WD로 학습하며 추론 시 제거한다.

### 3.3 누락된 attention pooling의 구체적 대체안

[att.py](https://github.com/zhengxuJosh/C2VKD/blob/fe1ab3d6f815969058451c221229a744fe872a47/models/deeplabv3plus/att.py)에
`AttentionPool2d(16,2048,32,512)`가 있고 train.py는 `/pretrain_model0.pth`를 적재한다.
그러나 해당 모듈 소스·checkpoint·사전학습 절차는 확인한 저장소에 없다.
**논문을 읽는 것만으로 이 학습된 가중치를 복원할 수는 없다.**

실행 가능한 별도 재구현안으로 다음을 지정한다.

1. 논문이 인용한 [DenseCLIP attention pooling](https://github.com/raoyongming/DenseCLIP/blob/5eda47f01a1011286c98425554cc96e7df6eb759/segmentation/denseclip/models.py)의
   GAP 토큰 + 공간 토큰 + MHSA 구조를 사용한다. 폭 2048, head 32, 출력 512, dropout 0.
2. [OpenAI CLIP RN101](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py)의
   `visual.attnpool` 가중치만 가져온다. Q/K/V 및 출력 projection의 key/shape를 엄격 검사한다.
   기대 위치 embedding은 7×7+global이며 공간 부분만 bilinear/False로 16×16에 보간한다.
3. Pool 전체를 freeze/eval하고 CIRKD teacher의 layer4를 resize한 feature에 적용한다.
   CLIP의 CNN backbone, text encoder, text prompt, CLIP 이미지 전처리는 사용하지 않는다.
4. 원본 checkpoint 예상 SHA-256은 공식 URL에 명시된
   `8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599`이다.
   **실제 파일은 아직 다운로드·적재 검증하지 않았다.** 파생 pool 가중치도 별도 hash를 기록한다.

이 선택은 **C2VKD 저자가 CLIP RN101을 사용했다는 발견이 아니다.** 공개된 모듈 규격에 맞춘
대체 가중치 제안이며, CIRKD feature와 CLIP pool의 표현 분포가 맞는지도 미검증이다.
추가 사전학습 자료를 도입하므로 기존 6개 방법과 완전히 같은 사전학습 조건으로 취급하지 않는다.
`C2VKD* (CLIP-pool)`로 별도 보고한다. 임의 초기 pool을 고정한 약한 기준선이나 linguistic 항을
뺀 변형을 C2VKD 전체 기법으로 이름 붙이지 않는다.
원본 pool을 추후 확보하면 별도 revision으로 교체하고 그 가중치의 출처·사전학습 조건도 점검한다.

### 3.4 누락된 PDD의 수식 기반 정의

각 유효 픽셀에서 softmax 확률 `s,t`와 정답 class `y`를 사용한다.
`s_nt`, `t_nt`는 **y 이외 class의 확률 합**이다. 마지막 class를 무조건 target으로 사용하지 않는다.

```text
s2 = [s_y, sum_{c != y} s_c]
t2 = [t_y, sum_{c != y} t_c]
q2 = (t2 + [1,0]) / 2
PDD = mean_valid_pixels(0.5*s2[0]*log(s2[0]/q2[0])
                     + 0.5*s2[1]*log(s2[1]/q2[1]))
```

이는 Eq.8의 target/non-target 이진 확률과 Eq.9의 student→teacher+GT 방향을 구현하는
**보완 정의**다. 두 비율이 같은 0.5이므로 분모를 확률분포로 정규화해도 원문 식과의 차이는
`0.5*log(2)`라는 상수다(수치 clamp가 없는 경우). 따라서 student gradient는 같다.
확률 0의 수치 문제에는 eps=1e-8로 log 입력을 보호하고 q2를 다시 정규화한다.
`s_nt`는 `1-s_y`의 상쇄 대신 non-target 확률을 직접 합한다.
클래스별 non-target 조건부 KL을 추가하는 표준 DKD 변형은 넣지 않는다.

Logits resize는 공통과 같은 512×512, bilinear/True. Ignore=-1은 gather 전 제거하며
유효 픽셀 평균을 사용한다. Temperature 1, CE 추가 계수 0, 일반 logit KD 추가 계수 0이다.

### 3.5 공개된 VLFD의 계산 기준

```text
L = PDD + 0.1*G + 0.1*P + 0.5*Linguistic
G = mean_{b,c,h,w} pT[c,h,w]*(log pT[c,h,w] - log pS[c,h,w])
Linguistic = mean_{b,d} qT[d]*(log qT[d] - log qS[d])
```

`p`는 visual feature의 채널 2048개에 대한 softmax, `q`는 linguistic 512차원 softmax다.
두 항 모두 temperature 1이며 **class/channel 합이 아닌 전체 원소 평균**이다.
따라서 `KLDivLoss(reduction='mean',log_target=True)`와 같고 `batchmean`으로 바꾸지 않는다.
Teacher target은 detach한다.

Patch 항은 공개 코드대로 32×32 visual feature를 `[B*1024,2048]`로 펼친다.
각 행을 L2 norm(eps=1e-8)으로 나눈 뒤 그 행의 채널 평균을 뺀다.
`M = X @ X.T / (2048-1)`로 정의하고 `P=MSEmean(MS,MT)`를 사용한다.
**공개 코드에는 서로 다른 이미지의 위치 쌍도 포함**된다. 이를 이미지별 Gram으로
조용히 바꾸지 않는다. GT ignore mask는 이 feature 항들에 적용하지 않는다.
작은 수치가 나올 수 있지만 유한하다는 이유만으로 임의 배수를 추가하지 않는다.

메모리는 행 256개 단위의 exact chunk로 줄이고 모든 쌍의 squared-error 합을
`(B*1024)^2`로 나눈다. 역전파까지 메모리를 제한하려면 chunk별 checkpoint/recompute를
사용한다. 단순히 scalar들을 더해 전체 계산 그래프를 저장하면 메모리가 줄지 않을 수 있다.
본 비교는 단일 GPU 실제 batch 16이며 microbatch/gradient accumulation은 이미지 간 관계를 보존하지 않는다.

[C2VKD JSON](configs/c2vkd_method_protocol_v1.json)에 목적함수 예외와 추가 가중치 출처를 함께 기록했다.

## 4. 구현 시 통과할 검사와 실험 순서

1. 공통 teacher·student 및 C2 대체 pool의 파일 bytes/hash·key/shape 적재를 검사한다.
   CIRKD 반환 feature 대신 필요한 backbone layer를 명시적으로 추출한다.
2. 고정 crop에서 각 loss를 따로 기록하고 student/adapter로 gradient가 전달되는지 확인한다.
   FSKD query 평균 attention, soft-rank, C2 PDD의 target/ignore 축을 작은 합성 입력으로 검사한다.
3. 작은 입력에서 chunk/non-chunk loss와 gradient가 일치하는지 확인한다.
   C2 PDD 정규화 전후 gradient 일치, 원 코드와 공개 항 reduction 일치도 확인한다.
4. Teacher/pool parameter·BN 불변, 공통 student 초기화·증강 일치, full-state resume를 검사한다.
   모든 loss의 raw/weighted 값과 encoder gradient norm, 추가 파라미터·메모리·시간을 기록한다.
   방법 JSON hash, adapter state, optimizer, RNG, sampler, 완료 step을 checkpoint에 저장하고,
   C2 후보는 원본·변환 pool hash도 기록한다. 환경의 PyTorch·CUDA·torchsort 버전도 고정한다.
5. FSKD는 고정값 1개로 smoke → 2k → 10k → 80k로 진행한다. C2 재구현 후보도 같은
   학생 학습 예산으로 별도 진행할 수 있지만 CLIP 사용·CE 예외를 결과표에 반드시 표시한다.
   낮은 초기 mIoU만으로 baseline을 제거하지 않는다.

이 문서는 **방법 연결·첫 시도 계수·누락 구현의 처리 방안까지 정한 상태**다.
저자의 비공개 Cityscapes 설정/원본 pool을 확보한 상태나 성능 검증 완료 상태는 아니다.
원본 설정을 추가 확보하거나 smoke에서 오류가 발견되면 변경 이유를 기록하고 버전을 올린다.
성능을 본 뒤 더 유리한 해석으로 손실을 바꿔 같은 run으로 이어가지 않는다.
