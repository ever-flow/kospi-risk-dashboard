# 코스피 리스크 스코어 대시보드

코스피의 향후 6개월 수익률을 예측하는 머신러닝 기반 리스크 스코어 대시보드입니다.

## 🎯 주요 기능

- **실시간 리스크 스코어**: 0-100 스케일의 직관적인 위험도 지표
- **22개 기술적 지표**: 모멘텀, 변동성, 밸류에이션, 자금흐름 등 종합 분석
- **동적 가중치**: 머신러닝 기반 시간별 가중치 변화 추적
- **인터랙티브 차트**: Plotly 기반 고품질 시각화
- **과거 성과 비교**: 1주일/1개월/6개월 전 대비 변화

## 📊 리스크 스코어 해석

- **80-100**: 매우 높음 (위험) - 포지션 축소 고려
- **60-80**: 높음 - 신중한 투자 필요
- **40-60**: 중간 - 균형잡힌 포트폴리오 유지
- **20-40**: 낮음 - 점진적 포지션 확대 고려
- **0-20**: 매우 낮음 (안전) - 적극적 투자 고려

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/kospi-risk-dashboard.git
cd kospi-risk-dashboard

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 대시보드 실행

```bash
streamlit run app/dashboard.py
```

브라우저에서 `http://localhost:8501`로 접속하세요.

## 📁 프로젝트 구조

```
kospi-risk-dashboard/
├── app/
│   ├── dashboard.py      # 메인 Streamlit 대시보드
│   ├── data_collector.py # 데이터 수집 모듈
│   ├── risk_engine.py    # 리스크 스코어 계산 엔진
│   └── __init__.py       # 패키지 초기화
├── requirements.txt      # Python 의존성
├── README.md             # 프로젝트 문서
├── .gitignore            # Git 무시 파일
└── docs/                 # 추가 문서
    ├── user_guide.md   # 사용자 가이드
    └── technical.md    # 기술 문서
```

## 🔧 주요 모듈

### `app/dashboard.py`
- Streamlit 기반 웹 대시보드
- 인터랙티브 차트 및 사용자 인터페이스
- 실시간 데이터 업데이트

### `app/data_collector.py`
- Yahoo Finance API 연동
- KRX 데이터 수집
- 캐시 시스템으로 효율적 데이터 관리

### `app/risk_engine.py`
- 22개 기술적 지표 계산
- 머신러닝 기반 동적 가중치 계산
- 리스크 스코어 정규화 및 스무딩

## 📈 사용되는 지표

### 모멘텀 지표
- RSI (Relative Strength Index)
- MACD Divergence
- Bollinger Band Position
- 20일 모멘텀
- 50일 이동평균 대비 위치

### 변동성 지표
- 20일 변동성
- 변동성 비율 (vs S&P500)

### 수익률 지표
- 20일 수익률
- 상대 수익률 (vs S&P500)

### 거시경제 지표
- 원달러 환율 변화
- 미국 장단기 금리차

### 원자재 지표
- 구리/금 비율
- 원유 수익률 및 변동성

### 밸류에이션 지표
- PER (주가수익비율)
- PBR (주가순자산비율)

### 시장 심리 지표
- VIX (공포 지수)

### 자금 흐름 지표
- 외국인 자금흐름
- 기관 자금흐름
- 개인 자금흐름

### 채권 지표
- 중기국채 ETF 수익률 및 변동성

## ⚙️ 설정

### 캐시 설정
데이터는 1시간마다 자동 갱신되며, 수동으로 강제 업데이트할 수 있습니다.
네트워크 오류로 새 데이터를 받지 못한 경우에는 기존 캐시가 자동으로 사용됩니다.

### API 설정
- Yahoo Finance: 무료 API 사용
- KRX: pykrx 라이브러리 사용

### 하이퍼파라미터 조정
리스크 스코어 계산에 사용되는 하이퍼파라미터는 다음과 같이 최적화된
기본값으로 설정되어 있습니다.

```python
{
    "roll": 126,
    "corr": 504,
    "step": 3,
    "model": "elastic",
    "alpha": 1.0,
    "future_days": 126,
    "smooth": 20,
}
```

필요 시 `RISK_PARAMS` 환경변수에 JSON 형식으로 값을 전달하거나
`RiskScoreEngine` 초기화 시 파라미터 사전을 전달하여 조정할 수 있습니다.


예시:

```bash
export RISK_PARAMS='{"roll":252,"model":"ridge"}'
```

## 🔍 기술적 세부사항

### 리스크 스코어 계산 과정
1. **지표 계산**: 22개 기술적 지표 계산
2. **정규화**: 백분위수 및 Z-스코어 정규화
3. **가중치 계산**: Ridge/Lasso/ElasticNet 회귀 기반 동적 가중치
4. **스코어 계산**: 가중합으로 원시 리스크 스코어 계산
5. **정규화**: 5년 롤링 윈도우 백분위수로 0-100 스케일 변환
6. **스무딩**: 지수가중평균으로 노이즈 제거

### 성능 최적화
- Streamlit 캐싱으로 빠른 로딩
- 효율적인 데이터 구조 사용
- 메모리 최적화된 계산

## 📊 호스팅 옵션

### Streamlit Cloud (추천)
1. GitHub에 코드 업로드
2. [Streamlit Cloud](https://streamlit.io/cloud) 접속
3. 저장소 연결 후 자동 배포

### Heroku
```bash
# Procfile 생성
echo "web: streamlit run app/dashboard.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Heroku 배포
heroku create your-app-name
git push heroku main
```

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## ⚠️ 주의사항

- 리스크 스코어는 참고용이며, 투자 결정의 유일한 기준이 되어서는 안 됩니다
- 과거 성과가 미래 수익을 보장하지 않습니다
- 다른 분석 도구와 함께 종합적으로 판단하세요

## 📄 라이선스

MIT License

## 🤝 기여

이슈 및 풀 리퀘스트를 환영합니다!

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

