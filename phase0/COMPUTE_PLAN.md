# 로컬 및 H200 연산 계획

## 로컬 워크스테이션

다음 작업은 로컬에서 진행합니다.

- Phase 0의 파일 무결성, metadata, mask, metric 감사
- 저장소 단위 테스트와 smoke test
- 정성 결과 확인과 보고서 생성
- Phase 0.5 Flowers feature caching과 frozen-probe 진단

현재 Apple M2 Max / 32GB 환경에서 확인한 결과는 다음과 같습니다.

- 단위 테스트 14개는 1초 이내에 완료
- 약 22MB인 체크포인트 3개의 deep audit은 수 초 이내에 완료
- 이미지–마스크 8,189쌍의 원본 해상도 감사는 약 2분에 완료
- 합성 입력 기반 batch 8 DeiT-Ti 12-block feature 추출은 CPU에서 초당 약
  272장 처리

마지막 처리량은 JPEG decode와 transform 시간을 포함하지 않지만, Phase 0는
로컬에서 충분히 수행할 수 있고 feature를 cache한 Phase 0.5 진단도 현실적이라는
점을 보여줍니다. 현재 Codex 실행 환경에서는 PyTorch MPS backend가 build되어
있지만 사용할 수 없으므로 CPU만으로도 성립하는 계획을 기준으로 합니다.

## H200

다음 작업부터 H200을 사용합니다.

- Phase 1 Pet에서 Vanilla/KD/LG/ALG/iBKD 분류 encoder 재학습
- 여러 encoder-training seed 반복
- end-to-end decoder fine-tuning 반복
- 표준 multi-class semantic-segmentation benchmark
- dense teacher–student distillation 직접 학습

H200 실행은 로컬에서 검증된 동일 manifest와 config를 사용해야 합니다. 각 실행
요약에는 Git commit, config, seed, dependency 버전, checkpoint hash, dataset
hash를 반드시 기록합니다.
