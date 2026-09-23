# H200 2k v1 pack1 실패 점검

2026-09-24 후속 [v2 pack1 결과](../h200_screen2000_v2_pack1/RESULTS.md)에서 6개 모두 2k를 완료했습니다.
새 입력 검사에 batch26 sample4 `jena_000087_000019_gtFine_labelIds`의 ignore-only crop이 기록돼
아래 원인 진단이 확인됐습니다. 이 문서는 당시 v1 실패의 증거와 검사 범위를 그대로 보존합니다.

2026-09-23 사용자 제출 로그의 최종 집계는 **failed, 0/6 완료**입니다.
Vanilla·FSKD·LG 네 β 모두 25 update 후 26번째 batch 로딩에서 같은 예외로 멈췄습니다.
전체 val500은 시작하지 않았고 mIoU·선택 step은 null, LG β 선택은 pending입니다.
이 로그로 후보 우열이나 장기 학습 성능을 판단하지 않습니다.

원본은 Git에 넣지 않고 [추출·점검 기록](log_audit.json)을 보존합니다.

- 원본 로그: 65,008 bytes.
- 원본 SHA-256: `394751cdc9482d1302026266a8550735f2a12b17fcb7b79e44f0ece42e2c6793`.
- 실행 commit: `cd9c8b8c1af9e2d8774c6b0dc876478a27191116`.
- 프로토콜: `cityscapes_b0_screen2000_v1`.
- 설치·준비·실패한 6개 실행을 포함한 총 시간: 717.24초.

## 확인된 상태

| 조건 | 완료 update | 마지막 전체 loss | 마지막 CE | 결과 |
|---|---:|---:|---:|---|
| Vanilla | 25 | 1.837006 | 1.837006 | 입력 오류 |
| FSKD* | 25 | 3.530360 | 1.781192 | 입력 오류 |
| LG β=0.197479 | 25 | 1.943883 | 1.812570 | 입력 오류 |
| LG β=0.460784 | 25 | 2.053072 | 1.796798 | 입력 오류 |
| LG β=0.987394 | 25 | 2.203022 | 1.744305 | 입력 오류 |
| LG β=1.97479 | 25 | 2.581485 | 1.752979 | 입력 오류 |

6개 모두 step2→3 checkpoint 복원·재실행을 통과했습니다.
데이터 해시 및 첫 25 batch calibration 입력 검사도 통과했습니다.
500-step 안정성 구간은 도달하지 못했습니다. 예외는 NaN/OOM이 아닌 데이터 검증 코드에서
발생했으며 `timm`의 FutureWarning은 중단 원인이 아닙니다.
마지막 손실은 서로 다른 목적함수·배치의 초기 값이므로 성능 순위로 사용하지 않습니다.

## 원인과 수정

`training_data.PlannedDataset.__getitem__`의 v1 검사는 이미지 유한성 검사와
`not (label != -1).any()`를 같은 `Invalid transformed sample` 예외로 처리했습니다.
따라서 random crop 전체가 ignore label인 정상 입력도 거부했습니다.
CIRKD의 고정된 원본 `CSTrainValSet`에는 이 거부 조건이 없습니다.

로그에 실패 crop의 실제 mask나 이름은 없어 해당 이미지를 직접 확인하지는 못했습니다.
하지만 uint8 이미지 읽기·resize·유한한 mean 빼기로 구성된 경로와 공통 실패 지점은
ignore-only 검사 분기를 원인으로 가리킵니다. 합성 ignore-only crop으로 같은 예외를 재현했고,
수정 전 회귀 검사 3개가 이 조건에서 실패하는 것을 확인했습니다.

v2는 개별 ignore-only crop을 그대로 반환합니다. Smoke/calibration의 tensor 변환에 있던
동일한 거부 조건도 함께 수정했습니다. 기존 CE·logit KD는 배치 전체의 유효 픽셀만
선택하므로 loss 수식·정규화는 바꾸지 않았습니다. Feature guidance는 원래 배치 전체를 유지합니다.
이미지 NaN/Inf 검사와 배치 전체에 유효 라벨이 없을 때의 명시적 실패는 유지합니다.
Crop 재추출·샘플 제외·batch 축소를 하지 않으며 β grid, 초기화, 난수 순서, LR, controller,
평가 및 선별 규칙도 유지합니다.

[v2 명세](../../configs/b0_screen2000_v2.json)는 원래 v1 해시를 참조하고,
출력은 `/app/output/cityscapes_b0_screen2000_v2/`로 분리합니다.
입력 사전 검사를 32 batch까지 늘려 배치별 유효 픽셀과 ignore-only crop 이름을 기록합니다.
최초 25 batch의 calibration 해시 기준은 유지하며 이 사전 검사는 optimizer update를 하지 않습니다.

## 수정 검증과 다음 실행

- 관련 단위 검사: **53 passed, 1 CUDA-only skipped**.
- 합성 입력의 26번째 batch에 ignore-only crop을 넣고 직렬·thread 로더에서 27 update 완료.
- CE·logit KD가 유효 픽셀만 정규화하고 ignore-only sample logit의 gradient가 0인지 확인.
- CIRKD 원본과 ignore-only crop이 bitwise 일치하고 validation에서도 입력을 유지하는지 확인.
- 실제 CIRKD/NVIDIA 가중치, 합성 batch2·crop64에서 6개 방법 모두 6 update,
  3회 평가, step2→3 재개 검사 통과. Replay batch에 ignore-only sample을 1개 넣었으며,
  validation 2장 중 1장을 ignore-only로 만들어 나머지 1장의 8,192픽셀만 집계되는 것도 확인.

이 검사는 CPU·합성 입력으로 수행했습니다. 실제 실패 crop, v2 H200 2k 완료와 전체 val500은
아직 검증하지 않았습니다. [v2 실행 이슈 3개](../../H200_SCREEN2000_ISSUES.md)로 확인합니다.
v1 checkpoint는 코드·프로토콜 식별자가 달라 재개하지 않으며 모든 후보를 seed1부터 시작합니다.
Pack2·3 로그는 이번 첨부에 없지만 같은 데이터 로더를 사용하므로 동일 수정본을 적용합니다.
새 smoke 이슈 대신 32-batch CPU 입력 검사와 기존 내장 step2→3 재개 검사를 사용합니다.
