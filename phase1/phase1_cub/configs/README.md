# CUB-200-2011 설정

CUB 전용 본실험 프로토콜은 `cub200_b128_full_v1.json`에 고정했습니다. SHA-256은
`86e23457579935f68841bf068b613379e2f96d49e1196875064049a506058374`입니다.

현재 `cub200_b128_combined_smoke_v2.json`은 분류부터 frozen probe까지의 실행
경로만 검증하는 비과학적 smoke 계약입니다. 본실험 설정이 아니며, smoke metric을
방법·lambda·checkpoint 선택이나 논문 결론에 사용하는 것을 금지합니다.

실행 전 폐기한 v1의 canonical ALG warm-up 0 대신, v2는 유일한 ALG 조건을
`ALG-w20`으로 사전 고정합니다. v1의 내용은 Git 이력에만 남깁니다.

Full v1은 여섯 설정 전체를 하나의 계약으로 유지하면서 10시간 제한용 실행 shard
A/B만 나눕니다. 두 shard 사이에서 hyperparameter나 평가 규칙을 변경할 수
없습니다.
