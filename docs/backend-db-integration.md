# WattUp v9 DB와 백엔드 연동

## 연결 흐름

1. 앱은 `GET /api/charging-stations`에서 호환되는 주변 후보를 최대 10곳 가져옵니다. 현재 빈자리가 0대여도 목록에 남도록 `availableOnly=false`를 사용합니다.
2. 앱은 후보의 ID와 사용자 출발 위치를 `POST /api/predictions/batch`에 한 번 보냅니다. Spring Boot가 MySQL 상태 이력에서 입력을 만들고 기존 카카오 경로 클라이언트에서 후보별 도로 거리와 ETA를 받습니다.
3. Spring Boot는 v9에 `POST /internal/v3/arrival-predictions/batch`로 예측을 요청합니다. v9는 후보별 확률을 반환합니다. 앱은 충전소 ID로 주변 목록과 예측을 합칩니다.
4. 결합순에서는 `2 * distanceKm + 15 * (1 - probability)`를 계산해 낮은 순으로 표시합니다. 가까운 순과 확률순도 같은 목록에서 바꿀 수 있습니다. 도로 거리가 없거나 예측이 없으면 결합 점수를 비워 둡니다.

실제 구현은 [Backend 연동 브랜치](https://github.com/YU-AI-Service-Project/Backend/tree/feat/eta-v9-integration)를 사용합니다.

## 어떤 DB에서 무엇을 읽는지

| MySQL 데이터 | 사용하는 내용 |
|---|---|
| `charging_stations`, `chargers` | 공개·미삭제 후보, 호환 범위와 정격 출력, 충전소 위치 |
| `charger_status_observations` | 각 5분 시점까지 알려진 최신 상태와 충전 시작 시각, 공급자 상태 갱신 시각 |
| `charger_state_changes` | 최근 15분의 가용 상태 진입·이탈 횟수, 최근 30분의 충전 상태 진입·이탈 횟수 |
| `collection_runs` | 수집 완료 시각, 전체/변경분 수집 상태, 유효 구간 |

기존 Flyway 스키마를 사용합니다. Java의 `EtaHistoryRepository`는 DB UTC 시각으로 조인하고 `EtaFeatures`는 시간대 변수 계산에 한국 시각을 사용합니다. `charger_state_changes`의 새 관측을 수집 실행과 연결해, 완료 시점이 `asOf`를 넘는 사건은 포함하지 않습니다. 약 30분의 정상 이력을 확보한 후보가 예측 대상이 됩니다. 상태가 끊겼거나 알 수 없는 후보는 주변 목록에서 확인할 수 있고 예측 결과에 사유가 표시됩니다.

v9는 기존 숫자 입력에 아래 9개를 추가한 `daegu-eta-v9-events-37` 형식을 사용합니다. 모델의 37개 변수 중 예측 간격은 AI 서버가 계산합니다. 요청의 `features`에는 따라서 숫자 변수 36개와 품질 플래그 2개가 들어갑니다. 필드 전체는 [실행 예제](../example_ai_request.json)와 [모델 카드](../results/model_card.json)에 있습니다.

- `availabilityLosses15`, `availabilityGains15`
- `chargeStarts30`, `chargeEnds30`
- `availableAgeMin`, `availableAgeKnownFraction`
- `chargingElapsedMin`, `recentChargingCount`, `longChargingCount`

## 연동 환경 변수

| 프로세스 | 환경 변수 | 값 |
|---|---|---|
| AI | `WATTUP_INTERNAL_API_KEY` | Spring Boot와 공유하는 비밀 변수 |
| Spring Boot | `WATTUP_AI_ETA_BATCH_URL` | 로컬 예제 `http://127.0.0.1:8000/internal/v3/arrival-predictions/batch` |
| Spring Boot | `WATTUP_AI_ETA_FEATURE_VERSION` | `daegu-eta-v9-events-37` |
| Spring Boot | `WATTUP_AI_ETA_BEARER_TOKEN` | AI의 `WATTUP_INTERNAL_API_KEY`와 같은 값 |
| Spring Boot | `WATTUP_AI_HEALTH_URL` | 로컬 예제 `http://127.0.0.1:8000/health/ready` |
| Spring Boot | `WATTUP_KAKAO_REST_API_KEY` | 자동차 길찾기 권한을 가진 키 |
| Spring Boot | `WATTUP_DB_URL`, `WATTUP_DB_USER`, `WATTUP_DB_PASSWORD` | 기존 MySQL 연결 값 |

URL과 입력 버전을 함께 설정합니다. 원격 AI에 토큰을 보내는 구성에서는 Backend가 HTTPS를 요구합니다. AI가 정상이어도 경로를 계산할 수 없거나 상태 이력이 부족하면 앱은 해당 후보의 `probability=null`을 처리해야 합니다.

## 앱 요청 예제

```http
GET /api/charging-stations?latitude=35.834&longitude=128.751&radiusM=5000&limit=10&profile=ALL&availableOnly=false&sort=DISTANCE
```

위 응답의 `items[].stationId`를 사용합니다. 아래 ID는 형식 예제입니다.

```json
{
  "profile": "ALL",
  "mode": "ARRIVAL_ETA",
  "stationIds": ["KECO:STATION1", "KECO:STATION2"],
  "origin": { "latitude": 35.834, "longitude": 128.751 }
}
```

Spring Boot의 배치 결과에서는 `items[].prediction.probability`, `items[].prediction.roadDistanceM`, `items[].prediction.etaSeconds`, `items[].prediction.requestedArrivalAt`, `items[].prediction.status`를 사용합니다. 정렬 함수에 넣기 전에 도로 거리(m)를 1,000으로 나눕니다. 직선거리는 거리 탐색과 지도 표시용으로 사용할 수 있습니다. 점수와 확률은 별도 값입니다.
