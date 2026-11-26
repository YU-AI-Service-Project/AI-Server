# WattUp AI Server

충전소별 **예상 도착 시각에 차량 호환 충전기를 한 대 이상 이용할 확률**을 계산하는 FastAPI 서비스입니다. v9 모델 파일과 확률 보정 설정을 함께 제공합니다.

| 구성 | 위치 |
|---|---|
| 모델 서버와 독립 실행 예제 | [api_server.py](api_server.py), [example_ai_request.json](example_ai_request.json) |
| 모델 파일과 검증 결과 | [model_card.json](results/model_card.json), [metrics.json](results/metrics.json) |
| DB와 Spring Boot 연동 | [연동 가이드](docs/backend-db-integration.md) |
| 학습 및 평가 근거 | [모델 평가](docs/model-evaluation.md), [재학습](training/README.md) |
| 프론트 정렬 함수 | [rank_candidates.mjs](rank_candidates.mjs) |

## 실행

Python 3.11 이상에서 다음 명령을 실행합니다. PowerShell 기준입니다.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

같은 저장소의 다른 터미널에서 표본을 재생합니다.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/internal/v3/arrival-predictions/batch -ContentType application/json -InFile example_ai_request.json
```

서버에 `WATTUP_INTERNAL_API_KEY`를 주입하면 예측 요청에 `Authorization: Bearer <공유 토큰>`이 필요합니다. Spring Boot의 `WATTUP_AI_ETA_BEARER_TOKEN`에 같은 값을 주입합니다. 팀 서버에서는 배포 환경의 비밀 변수로 관리합니다. `WATTUP_MODEL_DIR`은 필요한 경우에만 모델 폴더의 절대 경로로 지정합니다.

## 검증

```powershell
.venv/Scripts/python.exe tests/verify_v9.py
node test_rank_candidates.mjs
```

v9는 대구의 상태 이력으로 학습한 개발 모델입니다. 실주행에서의 도착 시각과 전국 성능은 추가 평가가 필요합니다. 학습 원본 ZIP과 운영 비밀은 이 저장소에 포함하지 않습니다.
