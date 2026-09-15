# CUB 이미지 loader 본학습 결과

- [`damage_audit_v1/`](damage_audit_v1/RESULTS.md): 학습 전 augmentation 손상 감사
- [`full_v1_log_snapshot/`](full_v1_log_snapshot/RESULTS.md): L0/L1/L2 × guided
  4방법 validation-only 본실험과 loader 선택
- [`l2_guided_preliminary_full_seed1_log_snapshot_v1/`](l2_guided_preliminary_full_seed1_log_snapshot_v1/RESULTS.md):
  선택 L2의 guided 4방법 × encoder seed 1 official-test 예비 본실험

두 본실험 보고서는 H200 완료 로그에서 추출한 상태입니다. 결과 archive를 받으면
checkpoint와 원본 summary의 SHA-256을 독립 감사하고, 통과 시 log-snapshot 표기를
확정 결과 폴더로 승격합니다. 선행 실행 점검 및 실패 중간 산출물은 보존하지
않습니다.
