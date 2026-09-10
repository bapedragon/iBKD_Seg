# CUB-200-2011 설정

현재 본실험은 `cub200_r50_224_b128_full_v3.json`에 고정했습니다. SHA-256은
`e3faff49101a8cffc5d0836f2cf299177547cea5243715ce51cc288b743626dc`입니다.
Scratch ResNet-50/224 teacher와 최종 6설정×3seed 전체 수행, validation-only
선택 및 official-test 평가 규칙을 담습니다.

`cub200_r50_224_b128_guided_smoke_v3.json`은 teacher+LG+ALG-w20+iBKD 두 lambda의
2-epoch 분류→frozen probe 실행시간·메모리 smoke입니다. SHA-256은
`f1239e1533f40fce10bd2f5675c19284f26d91c804709b480de1dac745a2d1a4`입니다.
사전 고정한 전체 matrix를 변경하지 않는 조건으로 official test 경로도 검증하지만,
모든 smoke 수치는 비과학적입니다.

`cub200_r50_224_b64_b128_guided_smoke_v4.json`은 완료된 issue 722 Teacher를
release에서 내려받아 strict 검증한 뒤, LG·ALG-w20·iBKD λ=0.25/0.5의
2-epoch 분류→probe smoke를 student batch 128과 64에서 차례로 수행하는
비과학적 실행·용량 profile입니다. SHA-256은
`dd8e61c94f085096fed18615af217dd9b9e35bda0d79bdb19154d168c92ef211`입니다.
Batch 128이 잠긴 v3 주 실험이고 batch 64는 sensitivity 후보의 실행 가능성만
확인하므로, smoke 결과로 batch·방법·lambda를 선택하지 않습니다.

`cub200_r50_224_b128_b64_guided_seed1_full_v4.json`은 위 smoke 뒤 사전 고정한
full-epoch batch profile입니다. SHA-256은
`bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633`입니다.
Issue 722 Teacher 하나를 공유해 LG, ALG-w20, iBKD λ=0.25/0.5를 encoder seed 1,
300 epoch로 batch 128과 64에서 모두 학습한 뒤, encoder마다 5 probe seed × 3 LR ×
100 epoch를 수행합니다. Batch 128은 잠긴 v3 6방법×3seed 매트릭스에 편입 가능한
일부 셀이고, batch 64는 별도 sensitivity 결과입니다. 이 실행 하나만으로 최종
6방법×3seed 매트릭스가 완료됐다고 표시하지 않습니다.

`cub200_r50_224_b128_s23_b64_s2_guided_full_v5.json`은 seed-1 profile 결과를
확인한 뒤 고정한 후속 범위입니다. Batch 128은 원래 v3에서 확정했던 encoder seed
2·3을 이어서 수행해 seed `[1,2,3]`을 완성하고, batch 64는 encoder seed 2만
추가해 seed `[1,2]`의 탐색적 sensitivity로 보고합니다. Guided 네 방법의 학습과
probe 값은 v3/v4에서 변경하지 않습니다. Full config SHA-256은
`f3531c648f65e6f51e48bbeda7ad38b1fc5931d88e04b01c97c6ff71aad437b9`입니다.

`cub200_r50_224_b128_s23_b64_s2_guided_smoke_v5.json`은 위 후속 범위의 정확한
세 `(batch, encoder seed)` 조합을 각각 2-epoch 분류와 probe seed 1 × LR 3개 ×
2 epoch로 점검하는 비과학적 smoke입니다. Batch 64 seed 3이나 seed 1 재실행은
계약에서 거부합니다. Smoke config SHA-256은
`6160cdcb19f2225e574bf9f397b1c99be27c95c006ba7dfc88d9e41344042162`입니다.

`cub200_r50_224_b128_direct_spatial_smoke_v1.json`은 issue 727의 batch-128
encoder seed 1 네 개를 입력으로 visible-part localization probe, spatial CKA,
attention–GT rollout을 검증하는 비과학적 smoke 계약입니다. Part target은
`14×14`, Gaussian `σ=1`, visible-only MSE이고 주 지표는 bbox 최대 변으로
정규화한 PCK@0.1입니다. CKA는 validation의 ResNet-50 layer3와 DeiT block
`0..11`, attention은 residual-aware rollout과 patch AP를 사용합니다. Official
test는 열지 않으며 smoke config SHA-256은
`55ac0598c11a4065f3b1416022e8fbb3de35b21cad94780ace2e2036d5430bc6`입니다.

`cub200_r50_224_b128_seed1_direct_spatial_full_v2.json`은 직접 공간정보 진단의
seed-1 본실험 계약입니다. 첫 v1 실행은 metric 학습과 official-test 접근 전에
CUB image `5007`의 out-of-frame visible part를 발견해 중단됐습니다. v2는 원본
좌표를 clipping하지 않고 `official visible ∩ image bounds`만 part loss·validation
선택·PCK에 사용하며, split별 제외 내역을 전부 기록합니다. 나머지 4방법, probe
seed 5개, LR 3개, 100 epoch, CKA와 attention 계약은 v1과 같습니다. SHA-256은
`90f7dc92b7e1ad27b6a4a4b68e91bb5e72ea021389304d87dda5950fac8e6017`입니다.

`cub200_r50_224_b128_seed2_3_direct_spatial_smoke_v2.json`은 issue 730의
batch-128 encoder seed 2·3, 총 8개 checkpoint에 seed-1 본실험과 같은 직접
공간정보 진단 v2 정의를 적용하기 전 실행 경로를 점검하는 비과학적 smoke
계약입니다. 클래스별 한 장의 train/validation subset, part-probe seed 1 × LR
3개 × 2 epoch, validation CKA와 attention만 사용하고 official test는 열지
않습니다. Seed-1 결과로 설정을 바꾸거나 smoke 값으로 방법·lambda를 선택하지
않습니다. SHA-256은
`bd71b02ebcca3240c7278819b4a34416a131914bd442887afcce6251a7e366d1`입니다.

## 대체된 v2 보존본

과거 CUB 본실험 프로토콜은 `cub200_b128_full_v2.json`에 고정했습니다. SHA-256은
`0cf751c28168872a4108274644f80dadc7466d5c1210995e7da3abfc0737e575`입니다.

현재 `cub200_b128_combined_smoke_v2.json`은 분류부터 frozen probe까지의 실행
경로만 검증하는 비과학적 smoke 계약입니다. 본실험 설정이 아니며, smoke metric을
방법·lambda·checkpoint 선택이나 논문 결론에 사용하는 것을 금지합니다.

실행 전 폐기한 v1의 canonical ALG warm-up 0 대신, v2는 유일한 ALG 조건을
`ALG-w20`으로 사전 고정합니다. v1의 내용은 Git 이력에만 남깁니다.

실행되지 않은 full v1은 두 컨테이너에서 teacher를 각각 재학습하는 설계였으므로
폐기했고 Git 이력에만 남깁니다. Full v2 guided shard는 v3 교체 전에 제출되어
H200 issue 716에서 완료됐지만, baseline shard는 실행하지 않습니다. 회수한 v2
결과는 최종 v3와 합치지 않고 `reports/legacy_resnet56_v2_guided/`에 보존합니다.
