# Phase 1 Oxford-IIIT Pet 12-way timing 요청

GitHub issue form의 각 필드에 아래 값을 입력합니다.

## 제목

```text
[Request]: 박철현 Phase1 Pet 12-way batch64/128 lambda timing smoke
```

## 사용자 ID

```text
bapedragon
```

## 실행할 코드의 GitHub 링크

```text
https://github.com/bapedragon/iBKD_Seg.git
```

## 고정 commit

```text
COMMIT_SHA_AFTER_PUSH
```

## 코드 실행 명령어

```text
bash phase1/scripts/run_timing.sh
```

## 사용할 이미지

```text
pytorch/pytorch:latest
```

## 사용 언어

```text
Python
```

## GPU 할당량(MIG 개수)

```text
7
```

## 요청 내용

```text
Oxford-IIIT Pet Phase 1 본 학습 전 runtime/memory timing smoke입니다.

- 데이터: 공식 trainval 3,680장 안에서 고정 split seed 2027로 train 2,940 / validation 740(품종별 20장)
- official test 3,669장: 이 timing에서는 접근하지 않음
- teacher: scratch CIFAR-style ResNet56, 32x32, batch 128, 실제 2 epoch
- student: scratch DeiT-Tiny/16, 224x224, seed 1, FP32
- student matrix: (Vanilla, KD, LG, ALG, iBKD lambda=0.25, iBKD lambda=0.5) x (batch 64, batch 128) = 12 tasks
- 각 student: 전체 train에서 실제 2 epoch + 매 epoch validation, 300-epoch 예상시간 환산
- 기록: epoch 시간(validation 포함), peak CUDA memory, 성공/OOM, 동일 student 초기화 hash, 동일 split hash
- 한 조합이 실패/OOM이어도 다음 조합을 계속 실행
- timing accuracy/checkpoint는 batch, lambda, method, checkpoint 선택이나 논문 결과에 사용하지 않음
- 2-epoch timing teacher checkpoint는 guided timing 경로 확인에만 공통 사용하며 본 학습에 재사용하지 않음
- segmentation probe와 official test 평가는 이 요청에 포함하지 않음

실행 시작 시 1 teacher + 12 students = 총 13 tasks인지 로그로 확인해 주세요. 완료 후 timing_summary.json의 300-epoch 환산시간과 540분 safety cap 기반 추천 분할을 보고 후속 full-run 이슈를 나눌 예정입니다.
```

## 정상 완료 확인 문구

```text
[TASK_COUNT] teacher=1 student=12 total=13 matrix=6_variants_x_2_batches
[CONTRACT_CHECK] same_student_initial_state=True same_validation_split=True official_test_accessed=False
[SEQUENCE_DONE] status=complete|complete_with_failures completed=.../13 failed=... summary=...
```

## 결과로 첨부할 파일

```text
/app/output/phase1_pet_12way_timing_v1/timing_summary.json
/app/output/phase1_pet_12way_timing_v1/timing_summary.csv
/app/output/phase1_pet_12way_timing_v1/sequence_status.json
```
