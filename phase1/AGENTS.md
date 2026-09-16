# Phase 1 실험 범위 경계

- `phase1_cub/`는 CUB classification, frozen probe, 직접 공간정보 진단을 소유합니다.
- `phase1_cub_Seg/`는 CUB mask를 정답으로 사용하는 직접 segmentation 실험을 소유합니다.
- 한쪽 실험의 로그·산출물·진입점을 정리할 때 다른 쪽 폴더를 삭제, 이동, 이름 변경하거나
  관련 커밋을 revert하지 않습니다. 사용자가 두 영역을 함께 지정한 경우에만 함께 변경합니다.
- “classification 폴더에서 segmentation 로그를 제거”하는 요청은 `phase1_cub/` 내부만
  정리한다는 뜻이며, 독립된 sibling 폴더인 `phase1_cub_Seg/`에는 적용하지 않습니다.
