# Cityscapes 데이터 준비와 H200 전달

2026-09-14 사용자가 내려받은 `leftImg8bit_trainvaltest.zip`과
`gtFine_trainvaltest.zip`을 사용합니다. 데이터는 Git에 올리지 않습니다.
현재 모델은 **DeepLabV3 → Segmenter**, 비교는 Vanilla/LG/ALG/iBKD이며,
val pixel accuracy를 우선 보고하고 같은 checkpoint의 mIoU를 함께 기록합니다.

## 로컬 검증 결과

2026-09-14 검증 통과. 로컬 데이터 루트는
`/Users/bagcheolhyeon/Documents/AAAI Paper/iBKD_Seg/data/cityscapes`입니다.

| 항목 | 확인 결과 |
|---|---|
| Train | 이미지·labelIds 정답 2,975쌍 |
| Val | 이미지·labelIds 정답 500쌍 |
| 해상도 | 전체 2048×1024 일치 |
| PNG 디코딩·라벨 범위·유효 정답·split 중복 | 전체 통과 |
| ZIP CRC | 이미지 ZIP 10,036개, 정답 ZIP 40,036개 파일 통과; macOS 부가 파일 포함 |
| 추출 용량 | 8,895,004,654 bytes, train/val 및 부가 정답·README·license |
| 준비 코드 및 기존 Cityscapes 테스트 | 14개 통과 |

전달 파일의 정확한 크기와 SHA-256은 다음과 같습니다. 서버에서 같은 값이면
사용자의 ZIP이 변형 없이 전달된 것입니다.

| 파일 | Byte size | SHA-256 |
|---|---:|---|
| leftImg8bit_trainvaltest.zip | 11,598,567,370 | `1927eb6450e29ebde0d611d30b874abe96747f5d092f6ec0cbbb6a3e1f6ce09d` |
| gtFine_trainvaltest.zip | 263,041,307 | `dbacde05ea136f3036aa24bfc87b5272f0d999e34380e10cbb919f7368448332` |

로컬 `manifest.json` SHA-256:
`d0fdc7bfe937c00088ac2a4d0742b0a3e857d5d81c5b7e92b914d277bacb39e6`.
전체 파일별 기록은 Git에서 제외한 로컬 `preparation.json`과 `manifest.json`에 보관합니다.

## 운영진에게 전달할 요청 문구

아래 요청과 함께 Downloads에 있는 **원본 ZIP 두 개**를 운영진이 지정한 방식으로 전달합니다.
이 문서는 요청 초안이며 운영진에게 자동 발송하지 않았습니다.

> 안녕하세요. GitHub 사용자 bapedragon입니다.
> H200에서 Cityscapes segmentation 실험에 사용할 데이터 두 파일을 전달드립니다.
>
> - leftImg8bit_trainvaltest.zip
> - gtFine_trainvaltest.zip
>
> 이슈로 생성되는 제 실행 컨테이너에서 이 데이터를 읽을 수 있도록 등록 부탁드립니다.
> 서버에 저장된 위치와 함께, **실행 컨테이너 내부에서 접근할 절대 경로**를 알려주세요.
> 가능하면 이후 반복 실행에서도 같은 데이터에 접근할 수 있게 부탁드립니다.
>
> ZIP 그대로 배치하시면 두 ZIP이 들어 있는 폴더 경로를 알려주시면 됩니다.
> 압축을 풀어 배치하시면 `leftImg8bit/`와 `gtFine/`이 바로 아래에 있는 데이터 루트 경로를
> 알려주세요. 학습에는 train 2,975장, 검증에는 val 500장과 labelIds 정답을 사용합니다.

Mac 다운로드 완료와 H200 데이터 배치는 별도 단계입니다. 현재 서버 경로는 아직 없습니다.
[H200 운영 안내](https://github.com/Aerodrone-H200/gpu-request/blob/main/README.md)는
이슈로 컨테이너를 생성하고 코드 저장소를 clone하는 방식을 설명하며,
사용자의 컨테이너 직접 접속은 지원하지 않는다고 명시합니다.

## ZIP에서 안전하게 준비하는 명령

저장소 루트에서 실행합니다. 프로젝트 의존성이 설치된 Python을 사용합니다.
로컬에서 실행한 명령은 다음과 같습니다.

```bash
PYTHONPATH=src .venv/bin/python -m ibkd_seg.cityscapes.prepare \
  --zip-dir /Users/bagcheolhyeon/Downloads \
  --data-dir data/cityscapes
```

H200에 ZIP을 배치한 경우 아래 첫 번째 경로를 **운영진이 알려준 실제 컨테이너 경로로 변경**합니다.
`/app/scratch/cityscapes`는 해당 실행의 작업 폴더 예시이며 영구 공유 경로로 가정하지 않습니다.

```bash
export CITYSCAPES_ZIP_DIR="/운영진이/안내한/ZIP/폴더"
python -m pip install -e .
python -m ibkd_seg.cityscapes.prepare \
  --zip-dir "$CITYSCAPES_ZIP_DIR" \
  --data-dir /app/scratch/cityscapes
```

준비 명령은 다음을 확인합니다.

1. 두 ZIP의 train/val 개수를 먼저 검사하고 byte size·SHA-256을 기록합니다.
2. ZIP의 모든 파일을 끝까지 읽어 CRC를 검사합니다. Test 파일은 CRC만 확인하며 추출·학습·평가하지 않습니다.
3. 공식 폴더 구조와 상위에 패키지 폴더가 하나 더 있는 구조를 모두 처리하고 `__MACOSX`를 제외합니다.
4. Train/val 이미지와 정답 파일을 추출하고 원본 README·license를 `source_info/`에 보존합니다.
5. 이미지 3,475장과 대응 labelIds 정답 전체를 디코딩해 2048×1024 해상도, labelId 범위,
   유효 픽셀, split 중복 여부를 검사합니다. 파일별 byte size·SHA-256도 저장합니다.
6. 기존 파일과 내용이 같으면 재사용합니다. 다른 내용이면 덮어쓰지 않고 오류로 중단합니다.

정상 완료 시 `[CITYSCAPES_DATA_READY] train=2975 val=500`이 출력됩니다.
데이터 루트에 `manifest.json`(학습 이미지·정답 목록과 해시),
`preparation.json`(원본 ZIP 해시, CRC 및 추출 기록)이 생성됩니다.
현재 ZIP은 macOS 부가 파일과 상위 폴더를 포함하므로 기록한 SHA-256은
**전달받은 ZIP 자체의 식별값**입니다. 공식 ZIP의 MD5와 일치한다고 주장하지 않습니다.

```text
cityscapes/
├── leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
├── gtFine/{train,val}/<city>/*_gtFine_labelIds.png
├── source_info/
├── manifest.json
└── preparation.json
```

운영진이 이미 압축을 풀어 준 경우 ZIP 재추출 대신 기존 데이터 audit을 실행할 수 있습니다.
`cityscapes.run audit`은 데이터 검증이며 이전 모델의 학습을 실행하지 않습니다.

```bash
python -m ibkd_seg.cityscapes.run audit \
  --data-dir /운영진이/안내한/cityscapes \
  --output /app/output/cityscapes_manifest.json
```

## 본학습 전 남은 작업

H200 컨테이너 경로를 연결하고 동일한 데이터 검증을 수행합니다.
DeepLabV3/Segmenter 본학습에는 공개 가중치·전처리의 호환성 검증과 실제 데이터 학습 연결이
추가로 필요합니다. 합성 smoke의 임의 teacher 가중치로 성능 비교를 시작하지 않습니다.
기존 `cityscapes.run train/matrix`는 이전 자체 decoder pilot 명령입니다.

이 문서의 명령은 **데이터 준비·검증까지만 수행**하며 GPU 본학습을 자동 시작하지 않습니다.
