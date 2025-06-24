"""
코스피 리스크 스코어 대시보드 (최종 수정 버전)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as dt
from .data_collector import DataCollector
from .risk_engine import RiskScoreEngine

# 페이지 설정
st.set_page_config(
    page_title="코스피 리스크 스코어 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # 1시간 캐시
def load_risk_data(force_update=False):
    """리스크 데이터 로딩"""
    try:
        # 데이터 수집
        collector = DataCollector(force_update=force_update)
        data = collector.get_all_data()
        
        # 리스크 엔진
        engine = RiskScoreEngine()
        
        # 리스크 스코어 계산
        risk_scores, weights_history = engine.calculate_risk_score(
            data["prices"]["KOSPI"],
            data["prices"],
            data["per_pbr"],
            data["flows"]
        )
        
        # 지표 계산 (모든 지표 현황용)
        indicators = engine.calculate_indicators(
            data["prices"]["KOSPI"],
            data["prices"],
            data["per_pbr"],
            data["flows"]
        )
        
        return {
            "risk_scores": risk_scores,
            "weights_history": weights_history,
            "indicators": indicators,
            "market_data": data["prices"]["KOSPI"],
            "last_update": dt.datetime.now()
        }
        
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return None

def get_risk_level(score):
    """리스크 레벨 분류"""
    if score >= 80:
        return "매우 높음", "🔴", "risk-high"
    elif score >= 60:
        return "높음", "🟠", "risk-medium"
    elif score >= 40:
        return "중간", "🟡", "risk-medium"
    elif score >= 20:
        return "낮음", "🟢", "risk-low"
    else:
        return "매우 낮음", "🟢", "risk-low"

def create_risk_trend_chart(risk_scores, market_data):
    """리스크 스코어 추이 차트"""
    # 공통 인덱스
    common_idx = risk_scores.index.intersection(market_data.index)
    risk_data = risk_scores.reindex(common_idx).dropna()
    market_data = market_data.reindex(common_idx).dropna()
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('리스크 스코어', '코스피 지수'),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # 리스크 스코어 차트
    fig.add_trace(
        go.Scatter(
            x=risk_data.index,
            y=risk_data.values,
            mode='lines',
            name='리스크 스코어',
            line=dict(color='red', width=2),
            hovertemplate='날짜: %{x}<br>리스크 스코어: %{y:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 위험 구간 표시
    fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
    fig.add_hline(y=60, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=1)
    fig.add_hline(y=40, line_dash="dash", line_color="yellow", opacity=0.5, row=1, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
    
    # 코스피 지수 차트
    fig.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data.values,
            mode='lines',
            name='코스피 지수',
            line=dict(color='blue', width=2),
            hovertemplate='날짜: %{x}<br>코스피: %{y:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 레이아웃 설정
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="리스크 스코어 및 코스피 지수 추이",
        title_x=0.5
    )
    
    fig.update_xaxes(title_text="날짜", row=2, col=1)
    fig.update_yaxes(title_text="리스크 스코어", row=1, col=1)
    fig.update_yaxes(title_text="코스피 지수", row=2, col=1)
    
    return fig

def create_weights_chart(weights_history, selected_indicators, period_days):
    """가중치 시계열 차트"""
    # 기간 필터링
    end_date = weights_history.index[-1]
    start_date = end_date - pd.Timedelta(days=period_days)
    weights_filtered = weights_history.loc[start_date:end_date]
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, indicator in enumerate(selected_indicators):
        if indicator in weights_filtered.columns:
            fig.add_trace(
                go.Scatter(
                    x=weights_filtered.index,
                    y=weights_filtered[indicator],
                    mode='lines',
                    name=indicator,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'{indicator}<br>날짜: %{{x}}<br>가중치: %{{y:.3f}}<extra></extra>'
                )
            )
    
    fig.update_layout(
        title="지표별 가중치 시계열 변화",
        xaxis_title="날짜",
        yaxis_title="가중치",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_current_weights_chart(weights_history):
    """현재 가중치 차트"""
    current_weights = weights_history.iloc[-1].dropna()
    current_weights = current_weights[current_weights != 0]
    current_weights_abs = current_weights.abs().sort_values(ascending=True)
    
    # 상위 15개만 표시
    top_weights = current_weights_abs.tail(15)
    
    colors = ['red' if current_weights[idx] > 0 else 'blue' for idx in top_weights.index]
    
    fig = go.Figure(go.Bar(
        x=top_weights.values,
        y=top_weights.index,
        orientation='h',
        marker_color=colors,
        hovertemplate='지표: %{y}<br>가중치: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="현재 지표별 가중치 (절댓값 기준 상위 15개)",
        xaxis_title="가중치",
        yaxis_title="지표",
        height=600
    )
    
    return fig

def display_all_indicators(indicators):
    """모든 지표 현황 표시"""
    if indicators is None or indicators.empty:
        st.warning("지표 데이터를 불러올 수 없습니다.")
        return
    
    latest_indicators = indicators.iloc[-1]
    
    # 지표 카테고리 분류
    categories = {
        "모멘텀 지표": ["RSI", "MACD_Diff", "BB_Pos", "Mom20", "MA50_Rel"],
        "변동성 지표": ["Vol20", "Vol_Ratio"],
        "수익률 지표": ["Ret20", "RetDiff"],
        "거시경제 지표": ["KRW_Change", "YldSpr"],
        "원자재 지표": ["Cu_Gd", "Oil_Ret", "Oil_Vol"],
        "밸류에이션 지표": ["PER", "PBR"],
        "시장 심리 지표": ["VIX"],
        "자금 흐름 지표": ["F_Flow", "I_Flow", "D_Flow"],
        "채권 지표": ["IEF_Ret", "IEF_Vol"]
    }
    
    for category, indicators_list in categories.items():
        st.subheader(f"📊 {category}")
        
        cols = st.columns(min(len(indicators_list), 3))
        
        for i, indicator in enumerate(indicators_list):
            if indicator in latest_indicators.index:
                value = latest_indicators[indicator]
                if not pd.isna(value):
                    with cols[i % 3]:
                        st.metric(
                            label=indicator,
                            value=f"{value:.2f}",
                            help=f"최신 {indicator} 값"
                        )

def main():
    # 헤더
    st.markdown('<h1 class="main-header">📊 코스피 리스크 스코어 대시보드</h1>', unsafe_allow_html=True)
    
    # 수동 업데이트 버튼 (상단에 배치)
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 데이터 업데이트", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        force_update = st.checkbox("⚡ 강제 업데이트", value=False)
    
    # 사이드바 설정 (간소화)
    with st.sidebar:
        st.markdown("## 📈 리스크 스코어란?")
        st.markdown("""
        코스피의 향후 6개월 수익률을 예측하는 지표로, 0-100 스케일로 표현됩니다.
        
        - **0-20**: 매우 낮음 (안전)
        - **20-40**: 낮음
        - **40-60**: 중간
        - **60-80**: 높음
        - **80-100**: 매우 높음 (위험)
        """)
        
        st.markdown("---")
        
        # 주요 특징
        st.markdown("## 🔍 주요 특징")
        st.markdown("""
        - 22개 기술적 지표 활용
        - 머신러닝 기반 동적 가중치
        - 실시간 데이터 업데이트
        - 과거 성과 검증 완료
        """)
    
    # 데이터 로딩
    with st.spinner("데이터를 불러오는 중..."):
        data = load_risk_data(force_update=force_update)
    
    if data is None:
        st.error("데이터를 불러올 수 없습니다. 데이터 업데이트 버튼을 눌러 다시 시도해주세요.")
        return
    
    # 현재 리스크 스코어
    current_risk = data["risk_scores"].iloc[-1]
    risk_level, risk_emoji, risk_class = get_risk_level(current_risk)
    
    # 과거 데이터와 비교
    risk_1w = data["risk_scores"].iloc[-5] if len(data["risk_scores"]) >= 5 else current_risk
    risk_1m = data["risk_scores"].iloc[-20] if len(data["risk_scores"]) >= 20 else current_risk
    risk_6m = data["risk_scores"].iloc[-126] if len(data["risk_scores"]) >= 126 else current_risk
    
    # 메인 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
        st.metric(
            label=f"{risk_emoji} 현재 리스크 스코어",
            value=f"{current_risk:.1f}",
            delta=f"{risk_level}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        delta_1w = current_risk - risk_1w
        st.metric(
            label="1주일 전 대비",
            value=f"{risk_1w:.1f}",
            delta=f"{delta_1w:+.1f}"
        )
    
    with col3:
        delta_1m = current_risk - risk_1m
        st.metric(
            label="1개월 전 대비",
            value=f"{risk_1m:.1f}",
            delta=f"{delta_1m:+.1f}"
        )
    
    with col4:
        delta_6m = current_risk - risk_6m
        st.metric(
            label="6개월 전 대비",
            value=f"{risk_6m:.1f}",
            delta=f"{delta_6m:+.1f}"
        )
    
    # 마지막 업데이트 시간
    st.info(f"📅 마지막 업데이트: {data['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 리스크 스코어 추이", 
        "📊 모든 지표 현황",
        "⚖️ 지표별 가중치",
        "📊 가중치 시계열",
        "🔍 기술적 지표",
        "📚 해석 가이드"
    ])
    
    with tab1:
        st.subheader("📈 리스크 스코어 및 코스피 지수 추이")
        
        # 기간 선택
        period_options = {
            "6개월": 180,
            "1년": 365,
            "2년": 730,
            "3년": 1095,
            "전체": len(data["risk_scores"])
        }
        
        selected_period = st.selectbox("표시 기간 선택", list(period_options.keys()), index=2)
        period_days = period_options[selected_period]
        
        # 차트 생성 및 표시
        chart_data = data["risk_scores"].tail(period_days)
        market_chart_data = data["market_data"].tail(period_days)
        
        fig = create_risk_trend_chart(chart_data, market_chart_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 모든 지표 현황")
        display_all_indicators(data["indicators"])
    
    with tab3:
        st.subheader("⚖️ 현재 지표별 가중치")
        
        if data["weights_history"] is not None and not data["weights_history"].empty:
            fig = create_current_weights_chart(data["weights_history"])
            st.plotly_chart(fig, use_container_width=True)
            
            # 가중치 해석
            st.markdown("""
            **가중치 해석:**
            - 🔴 빨간색: 양수 가중치 (높을수록 위험 증가)
            - 🔵 파란색: 음수 가중치 (높을수록 위험 감소)
            - 절댓값이 클수록 해당 지표의 영향력이 큼
            """)
        else:
            st.warning("가중치 데이터를 불러올 수 없습니다.")
    
    with tab4:
        st.subheader("📊 지표별 가중치 시계열 변화")
        
        if data["weights_history"] is not None and not data["weights_history"].empty:
            # 지표 선택
            all_indicators = data["weights_history"].columns.tolist()
            selected_indicators = st.multiselect(
                "표시할 지표 선택 (최대 10개 권장)",
                all_indicators,
                default=all_indicators[:5] if len(all_indicators) >= 5 else all_indicators
            )
            
            # 기간 선택
            period_options = {
                "6개월": 180,
                "1년": 365,
                "2년": 730,
                "전체": len(data["weights_history"])
            }
            
            selected_period = st.selectbox("가중치 표시 기간", list(period_options.keys()), index=1, key="weights_period")
            period_days = period_options[selected_period]
            
            if selected_indicators:
                fig = create_weights_chart(data["weights_history"], selected_indicators, period_days)
                st.plotly_chart(fig, use_container_width=True)
                
                # 가중치 변화 분석
                st.subheader("📈 가중치 변화 분석")
                
                recent_weights = data["weights_history"].iloc[-30:][selected_indicators].mean()
                past_weights = data["weights_history"].iloc[-180:-30][selected_indicators].mean()
                
                weight_changes = recent_weights - past_weights
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**최근 가중치 증가 지표:**")
                    increasing = weight_changes.sort_values(ascending=False).head(3)
                    for indicator, change in increasing.items():
                        if change > 0:
                            st.write(f"• {indicator}: +{change:.3f}")
                
                with col2:
                    st.markdown("**최근 가중치 감소 지표:**")
                    decreasing = weight_changes.sort_values(ascending=True).head(3)
                    for indicator, change in decreasing.items():
                        if change < 0:
                            st.write(f"• {indicator}: {change:.3f}")
            else:
                st.warning("표시할 지표를 선택해주세요.")
        else:
            st.warning("가중치 데이터를 불러올 수 없습니다.")
    
    with tab5:
        st.subheader("🔍 주요 기술적 지표 현황")
        
        if data["indicators"] is not None and not data["indicators"].empty:
            latest = data["indicators"].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RSI", f"{latest.get('RSI', 0):.1f}")
                st.metric("변동성 (20일)", f"{latest.get('Vol20', 0):.1f}%")
            
            with col2:
                st.metric("VIX", f"{latest.get('VIX', 0):.1f}")
                st.metric("PBR", f"{latest.get('PBR', 0):.2f}")
            
            with col3:
                st.metric("수익률 (20일)", f"{latest.get('Ret20', 0):.1f}%")
                st.metric("원달러 변화", f"{latest.get('KRW_Change', 0):.1f}%")
        else:
            st.warning("기술적 지표 데이터를 불러올 수 없습니다.")
    
    with tab6:
        st.subheader("📚 리스크 스코어 해석 가이드")
        
        st.markdown("""
        ## 🎯 리스크 스코어 활용법
        
        ### 📊 스코어 구간별 의미
        - **80-100 (매우 높음)**: 향후 6개월 내 큰 하락 위험. 포지션 축소 고려
        - **60-80 (높음)**: 상당한 하락 위험. 신중한 투자 필요
        - **40-60 (중간)**: 보통 수준의 위험. 균형잡힌 포트폴리오 유지
        - **20-40 (낮음)**: 상대적으로 안전. 점진적 포지션 확대 고려
        - **0-20 (매우 낮음)**: 매우 안전한 구간. 적극적 투자 고려
        
        ### 🔍 주요 지표 설명
        - **RSI**: 과매수/과매도 상태 측정
        - **VIX**: 시장 공포 지수
        - **PBR**: 주가순자산비율 (밸류에이션)
        - **자금흐름**: 외국인/기관/개인 투자자 동향
        
        ### ⚠️ 주의사항
        - 리스크 스코어는 참고용이며, 투자 결정의 유일한 기준이 되어서는 안 됩니다
        - 다른 분석 도구와 함께 종합적으로 판단하세요
        - 과거 성과가 미래 수익을 보장하지 않습니다
        
        ### 📈 투자 전략 제안
        - **고위험 구간**: 현금 비중 확대, 방어주 선호
        - **중위험 구간**: 균형잡힌 포트폴리오 유지
        - **저위험 구간**: 성장주 비중 확대, 레버리지 고려
        """)

if __name__ == "__main__":
    main()

