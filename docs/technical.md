# 기술 문서

## 🏗️ 시스템 아키텍처

### 전체 구조
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Data Collector │    │   Risk Engine   │
│                 │───▶│                 │───▶│                 │
│ Yahoo Finance   │    │ - Caching       │    │ - Indicators    │
│ KRX (pykrx)     │    │ - Validation    │    │ - ML Models     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Plotly        │    │  Risk Scores    │
│   Dashboard     │◀───│   Charts        │◀───│  & Weights      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 데이터 플로우

### 1. 데이터 수집 (`data_collector.py`)

#### 데이터 소스
- **Yahoo Finance**: 주가, 지수, 원자재, 채권 데이터
- **KRX**: PER/PBR, 자금흐름 데이터

#### 캐싱 시스템
```python
# 캐시 구조
.cache/
├── price_data.pkl      # 가격 데이터 (1시간 TTL)
├── per_pbr_data.pkl    # PER/PBR 데이터 (1시간 TTL)
└── flow_data.pkl       # 자금흐름 데이터 (1시간 TTL)
```

#### 데이터 검증
- 결측값 처리: Forward Fill
- 이상값 탐지: IQR 방법
- 데이터 정합성: 공통 인덱스 생성

### 2. 지표 계산 (`risk_engine.py`)

#### 22개 기술적 지표

**모멘텀 지표 (5개)**
```python
"RSI": RSIIndicator(kospi, 14).rsi()
"MACD_Diff": MACD(kospi).macd_diff()
"BB_Pos": ((kospi - bb_lower) / bb_width * 100)
"Mom20": kospi.pct_change(20) * 100
"MA50_Rel": (kospi / kospi.rolling(50).mean() - 1) * 100
```

**변동성 지표 (2개)**
```python
"Vol20": kospi.pct_change().rolling(20).std() * sqrt(252) * 100
"Vol_Ratio": kospi_vol / sp500_vol
```

**수익률 지표 (2개)**
```python
"Ret20": kospi.pct_change(20) * 100
"RetDiff": kospi_ret20 - sp500_ret20
```

**거시경제 지표 (2개)**
```python
"KRW_Change": usdkrw.pct_change(20) * 100
"YldSpr": us10y - us2y
```

**원자재 지표 (3개)**
```python
"Cu_Gd": copper / gold
"Oil_Ret": oil.pct_change(20) * 100
"Oil_Vol": oil.pct_change().rolling(20).std() * sqrt(252) * 100
```

**밸류에이션 지표 (2개)**
```python
"PER": per_pbr["PER"]
"PBR": per_pbr["PBR"]
```

**시장 심리 지표 (1개)**
```python
"VIX": vix
```

**자금 흐름 지표 (3개)**
```python
"F_Flow": foreign_flow.rolling(20).sum()
"I_Flow": institution_flow.rolling(20).sum()
"D_Flow": individual_flow.rolling(20).sum()
```

**채권 지표 (2개)**
```python
"IEF_Ret": ief.pct_change(20) * 100
"IEF_Vol": ief.pct_change().rolling(20).std() * sqrt(252) * 100
```

#### 지표 정규화

**백분위수 정규화**
```python
def rolling_percentile(s, window, inverse=False):
    for i in range(len(s)):
        win = s[max(0, i-window+1):i+1]
        p = percentileofscore(win, s[i])
        return 100-p if inverse else p
```

**Z-스코어 정규화**
```python
def rolling_zscore(s, window):
    return (s - s.rolling(window).mean()) / s.rolling(window).std()
```

### 3. 머신러닝 모델

#### 동적 가중치 계산
```python
# 모델 선택
models = {
    "ridge": Ridge(alpha=alpha),
    "lasso": Lasso(alpha=alpha),
    "elastic": ElasticNet(alpha=alpha, l1_ratio=0.5)
}

# 롤링 윈도우 학습
for i in range(corr_window, len(data), step):
    X_train = indicators[i-corr_window:i]
    y_train = target_returns[i-corr_window:i]
    
    model.fit(X_train, y_train)
    weights[i] = model.coef_
```

#### 목표 변수
- **예측 대상**: 향후 126일(6개월) 수익률
- **방향 조정**: 음의 상관관계로 조정 (높은 스코어 = 높은 위험)

### 4. 리스크 스코어 계산

#### 원시 스코어 계산
```python
raw_risk = (normalized_indicators * weights).sum(axis=1)
```

#### 최종 정규화
```python
# 5년 롤링 백분위수
risk_normalized = raw_risk.rolling(1260, min_periods=252).apply(
    lambda x: percentileofscore(x, x[-1])
).clip(0, 100)

# 지수가중평균 스무딩
risk_final = risk_normalized.ewm(span=10).mean()
```

## 🎨 프론트엔드 (Streamlit)

### 컴포넌트 구조
```python
# 메인 레이아웃
st.set_page_config(layout="wide")

# 캐싱된 데이터 로딩
@st.cache_data(ttl=3600)
def load_risk_data():
    # 데이터 수집 및 계산
    
# 인터랙티브 차트
def create_risk_trend_chart():
    # Plotly 서브플롯
    
# 탭 기반 네비게이션
tab1, tab2, ... = st.tabs([...])
```

### 차트 라이브러리 (Plotly)
- **서브플롯**: 리스크 스코어 + 코스피 지수
- **인터랙티브**: 줌, 팬, 호버 정보
- **반응형**: 모바일 친화적 디자인

## 🚀 성능 최적화

### 캐싱 전략
1. **Streamlit 캐싱**: `@st.cache_data(ttl=3600)`
2. **파일 캐싱**: Pickle 기반 로컬 캐시
3. **메모리 최적화**: 필요한 컬럼만 로딩

### 데이터 처리 최적화
```python
# 벡터화 연산
indicators = pd.DataFrame({
    "RSI": rsi_values,  # 한 번에 계산
    "Vol20": vol_values,
    # ...
})

# 효율적인 롤링 계산
rolling_stats = data.rolling(window).agg(['mean', 'std'])
```

## 🔧 배포 고려사항

### Streamlit Cloud
```python
# streamlit/config.toml
[server]
maxUploadSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Docker 배포
```dockerfile
FROM python:3.9-slim

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY . /app
WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]
```

### 환경 변수
```bash
# 선택적 설정
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## 📈 모니터링 및 로깅

### 로깅 설정
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
```

### 성능 메트릭
- 데이터 로딩 시간
- 계산 처리 시간
- 메모리 사용량
- 캐시 히트율

## 🔒 보안 고려사항

### 데이터 보안
- API 키 환경변수 관리
- 민감한 정보 로깅 방지
- 입력 데이터 검증

### 웹 보안
- HTTPS 사용 권장
- CORS 설정
- 입력 sanitization

## 🧪 테스트 전략

### 단위 테스트
```python
def test_risk_engine():
    engine = RiskScoreEngine()
    # 테스트 데이터로 검증
    
def test_data_collector():
    collector = DataCollector()
    # API 응답 검증
```

### 통합 테스트
- 전체 파이프라인 테스트
- 데이터 품질 검증
- 성능 벤치마크

## 📊 확장 가능성

### 추가 기능
- 알림 시스템
- 포트폴리오 분석
- 백테스팅 기능
- API 엔드포인트

### 스케일링
- 데이터베이스 연동
- 마이크로서비스 아키텍처
- 실시간 스트리밍

