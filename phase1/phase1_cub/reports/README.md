# CUB-200-2011 결과 보고서

검증된 요약, manifest와 작은 정성 예시만 이 폴더에 보존합니다. 데이터셋,
checkpoint, feature cache와 원시 실행 결과는 Git에 포함하지 않습니다.

먼저 guided 결과 archive의 teacher checkpoint를 hash로 고정합니다. 이후 동일
teacher를 사용하는 baseline 결과까지 받은 뒤 초기 student state·split·config
hash를 교차 검사하고, 분류와 probe 결과를 하나의 6설정 보고서로 병합합니다.
