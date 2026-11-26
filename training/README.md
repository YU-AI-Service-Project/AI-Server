# v9 재학습과 재현

팀이 수집한 원본 ZIP `ev_data_1119.zip`을 로컬에서 준비합니다. SHA-256은 `results/selection_lock.json`의 `data_sha256`에 있습니다. ZIP에는 `export/charger.csv`, `export/status_event.csv`, `export/collect_log.csv`가 있어야 합니다.

저장소 최상위에서 실행합니다.

```powershell
$export = 'D:/data/ev_data_1119.zip'
python training/observation_builder.py $export
python training/event_features.py $export
python training/train_v9.py --source $export
python tests/verify_v9.py --source $export
```

상태 복원 배열은 `training/state_replay`, 새 상태 변화 입력은 `training/event_data`에 생성됩니다. 새 모델, 평가 결과, 새 실행 예제는 `training/retrained`에 저장됩니다. 원본 ZIP과 생성 배열은 Git에 올리지 않습니다. `training/reference`에는 같은 날짜의 기존 v8 비교 모델과 고정한 충전소 보류 목록을 제공합니다.

이 저장소의 스크립트로 다시 학습해 모델 파일의 SHA-256과 평가값이 공개 모델과 일치함을 확인했습니다.

기간을 새로 늘려 학습할 때는 `replay.py`와 `query_replay.py`의 날짜 분할 및 비교 기준을 갱신해야 합니다. 상태 데이터로 관측 시점의 입력만 구성하고, 최신 테스트 기간을 보류한 뒤 모델을 선택합니다.
