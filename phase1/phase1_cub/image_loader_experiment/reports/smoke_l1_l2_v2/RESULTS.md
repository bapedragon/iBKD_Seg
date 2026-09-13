# CUB loader pilot L1·L2 subset smoke 결과

판정: **PASS — L1·L2 결합 본실험 실행 가능**

H200 issue 750에서 L1 `l1_matched_weak`와 L2
`l2_conservative_spatial`의 운영용 subset smoke를 수행했습니다. 418.26초에
분류 `8/8`, segmentation/part 후보 각 `24/24`, 선택 각 `8/8`, CKA `96/96`,
attention `8/8`, 정성 PNG `32/32` gate를 모두 통과했습니다. Official test 접근은
0회였고 OOM, NaN, traceback 또는 비정상 종료는 관측되지 않았습니다. 최대 기록
CUDA allocated/reserved는 각각 11,912,255,488 / 15,881,732,096 byte입니다.

Smoke가 출력한 단순 선형 full 상한은 10시간 4분 19초입니다. 이 값은 2-epoch의
초기 guidance 비용과 준비 비용을 full epoch 전체에 보수적으로 확대하므로 그대로
예상시간으로 해석하지 않습니다. 동일 실행기의 L0 상한 대비 실제시간 비율
`12,254 / 17,883 = 0.6852`를 작업 분할에만 적용하면 L1·L2 예상 합계는 약
6시간 54분 6초입니다. 동일 H200에서 약 45% 이상 느려져야 10시간에 도달하므로
두 shard를 한 이슈에서 순차 실행하기로 했습니다.

2-epoch 성능 수치는 구현 확인용이며 loader·방법·lambda 선택이나 논문 결론에
사용하지 않습니다. 결합 본실험도 L1/L2 출력·cache·checkpoint를 분리하고 official
test를 열지 않습니다. L0/L1/L2 전체 archive를 회수·감사한 뒤에만 사전 고정한
validation Part PCK → frozen-seg mIoU → L0/L1/L2 순서로 loader를 선택합니다.
