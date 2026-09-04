# Phase 1 Oxford-IIIT Pet batch 128 full classification 요청

이 문서는 GitHub issue form에 사용자가 직접 입력하기 위한 텍스트입니다. Codex가
issue를 자동 생성하거나 제출하지 않습니다.

## 제목

```text
[Request]: 박철현 Phase1 Pet batch128 6설정 x 3seed full classification
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
54787f5c8598ccd5a7b04f794cd7ff099b2d9dd5
```

## 코드 실행 명령어

```text
bash phase1/scripts/run_full_b128.sh
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
Oxford-IIIT Pet Phase 1의 batch-size 128 전체 37-way 품종 분류 요청입니다.

- 고정 commit: 54787f5c8598ccd5a7b04f794cd7ff099b2d9dd5
- 데이터: official trainval 3,680장에서 고정 split seed 2027로 train 2,940 / validation 740(품종별 20장), official test 3,669장
- 실행 전 gate: 공식 archive byte/MD5/SHA-256, split ID, image-label-trimap 1:1 대응, RGB/trimap decode와 trimap 값 {1,2,3} 감사
- test split은 위 무결성 감사 외에 학습·epoch 선택·hyperparameter 선택에 사용하지 않음
- teacher: scratch CIFAR-style ResNet56, seed 1, batch 128, 300 epochs, validation macro Top-1 최고 checkpoint 선택
- student: scratch DeiT-Tiny/16, batch 128, 300 epochs, FP32
- student matrix: (Vanilla, KD, LG, ALG, iBKD lambda=0.25, iBKD lambda=0.5) x encoder seed (1,2,3) = 18 runs
- 총 task: teacher 1 + student 18 = 19
- 각 student는 validation macro Top-1 최고 epoch를 선택하고 동률이면 이른 epoch를 선택
- 선택 checkpoint를 새 모델에 strict-load하고 hash를 확인한 뒤 official test를 정확히 한 번 평가
- test 보고값: macro Top-1(주 지표), overall Top-1, Top-5; seed 원값과 mean +/- sample SD 모두 저장
- 같은 seed의 여섯 설정은 같은 초기 student state와 split을 사용하고 hash로 검증
- guided student 전부가 이 이슈에서 선택한 동일 teacher checkpoint를 사용
- 결과가 좋은 lambda나 batch만 사후 선택하지 않으며 batch128 profile의 여섯 설정을 전부 보고
- segmentation/frozen probe는 이 요청에 포함하지 않음
- student 한 개가 실패해도 나머지는 계속 실행하고, 하나라도 실패하거나 계약 검사가 실패하면 최종 status는 complete_with_failures 및 non-zero exit
- 출력 루트: /app/output/phase1_pet_full_b128_v1

timing 로그 기반 순수 학습 환산시간은 8h 19m 51s입니다. 최초 설치/다운로드, dataset audit, checkpoint당 1회 test 시간이 추가되므로 600분으로 요청합니다.

후속 frozen segmentation probe에 validation-selected encoder가 필요하므로 summary 로그만이 아니라 output 루트 전체, 특히 students/*/student_best_validation.pt 18개를 반드시 보존해 주세요.
```

## 정상 완료 확인 문구

```text
[TASK_COUNT] teacher=1 student=18 total=19 matrix=6_variants_x_3_seeds batch=128
[CONTRACT_CHECK] same_student_initial_state=True same_validation_split=True same_teacher=True test_once=True test_used_for_selection=False
[SEQUENCE_DONE] status=complete completed=19/19 failed=0 summary=/app/output/phase1_pet_full_b128_v1/classification_summary.json
```

## 결과로 보존할 파일

```text
/app/output/phase1_pet_full_b128_v1/dataset_audit.json
/app/output/phase1_pet_full_b128_v1/classification_summary.json
/app/output/phase1_pet_full_b128_v1/classification_summary.csv
/app/output/phase1_pet_full_b128_v1/sequence_status.json
/app/output/phase1_pet_full_b128_v1/teacher/*/teacher_best_validation.pt
/app/output/phase1_pet_full_b128_v1/teacher/*/summary.json
/app/output/phase1_pet_full_b128_v1/students/*/student_best_validation.pt
/app/output/phase1_pet_full_b128_v1/students/*/summary.json
```

batch 64 이슈 결과와 함께 회수한 뒤 두
`classification_summary.json`의 `teacher_model_state_sha256`가 동일한지 먼저
확인합니다.
