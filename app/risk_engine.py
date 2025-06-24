"""
코스피 리스크 스코어 계산 엔진 (원본 코드 기반)
"""

import math
import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, spearmanr
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import logging

warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rolling_percentile(s: pd.Series, window: int, inverse=False):
    """롤링 백분위수 계산"""
    arr, out = s.to_numpy(float), np.full_like(s, np.nan, float)
    m = max(1, int(window * 0.6))
    for i in range(len(arr)):
        if math.isnan(arr[i]):
            continue
        win = arr[max(0, i - window + 1): i + 1]
        win = win[~np.isnan(win)]
        if win.size >= m:
            p = percentileofscore(win, arr[i], "rank")
            out[i] = 100 - p if inverse else p
    return pd.Series(out, index=s.index)

def rolling_zscore(s: pd.Series, window: int):
    """롤링 Z-스코어 계산"""
    r = (s - s.rolling(window).mean()) / s.rolling(window).std()
    return r.replace([np.inf, -np.inf], np.nan)

class RiskScoreEngine:
    """코스피 리스크 스코어 계산 엔진"""

    DEFAULT_PARAMS = {
        "roll": 126,
        "corr": 504,
        "step": 3,
        "model": "elastic",
        "alpha": 1.0,
        "future_days": 126,
        "smooth": 20,
    }

    def __init__(self, params: dict | None = None):
        """엔진 초기화 및 하이퍼파라미터 설정"""
        best = self.DEFAULT_PARAMS.copy()

        # 환경변수로 전달된 JSON 파라미터 처리
        env_params = os.getenv("RISK_PARAMS")
        if env_params:
            try:
                best.update(json.loads(env_params))
            except Exception as e:
                logger.warning(f"RISK_PARAMS 파싱 실패: {e}")

        # 직접 전달된 파라미터가 있으면 적용
        if params:
            best.update(params)

        self.best_params = best
        
    def calculate_indicators(self, kospi, prices, per_pbr, flows):
        """기술적 지표 계산 (원본 calc_risk 함수 기반)"""
        logger.info("기술적 지표 계산 중...")
        
        # 공통 인덱스 생성
        idx = kospi.index.intersection(prices.index).intersection(per_pbr.index).intersection(flows.index)
        kospi = kospi.reindex(idx).ffill()
        prices = prices.reindex(idx).ffill()
        per_pbr = per_pbr.reindex(idx).ffill()
        flows = flows.reindex(idx).ffill()
        
        # 볼린저 밴드
        bb = BollingerBands(kospi)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_width = bb_upper - bb_lower
        bb_position = ((kospi - bb_lower) / bb_width.replace(0, np.nan) * 100).fillna(50)
        
        # 기술적 지표 계산
        indicators = pd.DataFrame({
            # 모멘텀 지표
            "RSI": RSIIndicator(kospi, 14).rsi(),
            "MACD_Diff": MACD(kospi).macd_diff(),
            "BB_Pos": bb_position,
            "Mom20": kospi.pct_change(20) * 100,
            "MA50_Rel": (kospi / kospi.rolling(50, 25).mean() - 1) * 100,
            
            # 변동성 지표
            "Vol20": kospi.pct_change().rolling(20, 10).std() * math.sqrt(252) * 100,
            "Vol_Ratio": (kospi.pct_change().rolling(20).std() * math.sqrt(252) / 
                         (prices["SP500"].pct_change().rolling(20).std() * math.sqrt(252)).replace(0, np.nan)),
            
            # 수익률 지표
            "Ret20": kospi.pct_change(20) * 100,
            "RetDiff": kospi.pct_change(20) * 100 - prices["SP500"].pct_change(20) * 100,
            
            # 환율 및 금리
            "KRW_Change": prices["USDKRW"].pct_change(20) * 100,
            "YldSpr": prices["US10Y"] - prices["US2Y"],
            
            # 원자재
            "Cu_Gd": prices["COPPER"] / prices["GOLD"],
            "Oil_Ret": prices["CRUDE_OIL"].pct_change(20) * 100,
            "Oil_Vol": prices["CRUDE_OIL"].pct_change().rolling(20).std() * math.sqrt(252) * 100,
            
            # 밸류에이션
            "PER": per_pbr["PER"],
            "PBR": per_pbr["PBR"],
            
            # 시장 심리
            "VIX": prices["VIX"],
            
            # 자금 흐름
            "F_Flow": flows["Foreign"].rolling(20, 10).sum(),
            "I_Flow": flows["Institution"].rolling(20, 10).sum(),
            "D_Flow": flows["Individual"].rolling(20, 10).sum(),
            
            # 채권
            "IEF_Ret": prices["IEF"].pct_change(20) * 100,
            "IEF_Vol": prices["IEF"].pct_change().rolling(20).std() * math.sqrt(252) * 100,
        }, index=idx).ffill()
        
        return indicators
    
    def normalize_indicators(self, indicators: pd.DataFrame, window: int) -> pd.DataFrame:
        """지표 정규화 (원본 코드 기반)"""
        logger.info("지표 정규화 중...")
        
        # 높을수록 위험한 지표와 낮을수록 위험한 지표 분류
        high_risk_indicators = [c for c in indicators.columns if c not in {"YldSpr", "F_Flow", "I_Flow", "D_Flow"}]
        low_risk_indicators = ["YldSpr", "F_Flow", "I_Flow", "D_Flow"]
        
        # 백분위수 정규화
        sc_pct = pd.concat({
            k: rolling_percentile(indicators[k], window) 
            for k in high_risk_indicators + low_risk_indicators
        }, axis=1)
        
        # 낮을수록 위험한 지표는 역변환
        sc_pct[low_risk_indicators] = 100 - sc_pct[low_risk_indicators]
        
        # Z-스코어 정규화
        sc_z = indicators.apply(rolling_zscore, window=window).clip(-3, 3) * 10 + 50
        
        # 결합
        sc = pd.concat([sc_pct, sc_z.add_suffix("_Z")], axis=1)
        
        # MACD 특별 처리
        macd_max = indicators["MACD_Diff"].abs().rolling(window, int(window * 0.6)).max()
        sc["MACD_Sc"] = ((indicators["MACD_Diff"] / macd_max).fillna(0) * 50 + 50).clip(0, 100)
        
        return sc
    
    def calculate_target_returns(self, kospi: pd.Series, future_days: int) -> pd.Series:
        """목표 수익률 계산"""
        logger.info("목표 수익률 계산 중...")
        return kospi.pct_change(future_days).shift(-future_days)
    
    def calculate_dynamic_weights(self, indicators: pd.DataFrame, target_returns: pd.Series, 
                                corr_window: int, step: int, model_type: str, alpha: float) -> pd.DataFrame:
        """동적 가중치 계산 (원본 코드 기반)"""
        logger.info("동적 가중치 계산 중...")
        
        weights = pd.DataFrame(index=indicators.index, columns=indicators.columns, dtype=float)
        
        for i in range(corr_window, len(indicators), step):
            win = indicators.index[i - corr_window: i]
            train_X = indicators.loc[win]
            train_y = target_returns.loc[win]
            train_set = pd.concat([train_X, train_y.rename("target")], axis=1).dropna()
            
            if len(train_set) < int(corr_window * 0.8):
                continue
            
            # 모델 선택
            model_obj = {
                "ridge": Ridge(alpha=alpha),
                "lasso": Lasso(alpha=alpha, max_iter=10000),
                "elastic": ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000),
            }[model_type]
            
            # 모델 학습
            model_obj.fit(train_set[indicators.columns], train_set["target"])
            coef = model_obj.coef_
            
            # 계수 정규화 및 방향 조정
            if np.abs(coef).sum() > 1e-6:
                raw_train = (train_set[indicators.columns] * coef).sum(axis=1)
                if raw_train.var() > 0 and train_set["target"].var() > 0:
                    corr_train, _ = spearmanr(raw_train, train_set["target"])
                    if not math.isnan(corr_train) and corr_train > 0:
                        coef = -coef  # 음의 상관관계로 조정
                weights.iloc[i] = coef / np.abs(coef).sum()
        
        # 전진 채우기
        weights = weights.ffill().fillna(0)
        return weights
    
    def calculate_raw_risk_score(self, indicators: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
        """원시 리스크 스코어 계산"""
        logger.info("원시 리스크 스코어 계산 중...")
        return (indicators * weights).sum(axis=1)
    
    def normalize_risk_score(self, raw_risk: pd.Series, window: int = 1260, smooth: int = 10) -> pd.Series:
        """리스크 스코어 정규화 (0-100 스케일) - 원본 코드 방식"""
        logger.info("리스크 스코어 정규화 중...")
        
        # 5년 롤링 윈도우로 백분위수 계산 (원본 코드 방식)
        risk_normalized = raw_risk.rolling(window, min_periods=252).apply(
            lambda x: percentileofscore(x, x[-1]), raw=True
        ).clip(0, 100)
        
        # 지수 가중 평균으로 스무딩
        risk_smoothed = risk_normalized.ewm(span=smooth).mean()
        
        return risk_smoothed
    
    def calculate_risk_score(self, kospi, prices, per_pbr, flows):
        """전체 리스크 스코어 계산 프로세스"""
        try:
            # 1. 지표 계산
            indicators = self.calculate_indicators(kospi, prices, per_pbr, flows)
            
            # 2. 지표 정규화
            normalized_indicators = self.normalize_indicators(indicators, self.best_params["roll"])
            
            # 3. 목표 수익률 계산
            target_returns = self.calculate_target_returns(kospi, self.best_params["future_days"])
            
            # 4. 동적 가중치 계산
            weights = self.calculate_dynamic_weights(
                normalized_indicators, target_returns,
                self.best_params["corr"], self.best_params["step"],
                self.best_params["model"], self.best_params["alpha"]
            )
            
            # 5. 원시 리스크 스코어 계산
            raw_risk = self.calculate_raw_risk_score(normalized_indicators, weights)
            
            # 6. 리스크 스코어 정규화
            risk_scores = self.normalize_risk_score(raw_risk, smooth=self.best_params["smooth"])
            
            return risk_scores, weights
            
        except Exception as e:
            logger.error(f"리스크 스코어 계산 중 오류: {e}")
            raise

if __name__ == "__main__":
    # 테스트용 코드
    from app.data_collector import DataCollector
    collector = DataCollector()
    data = collector.get_all_data()
    
    engine = RiskScoreEngine()
    risk_scores, weights = engine.calculate_risk_score(
        data["prices"]["KOSPI"],
        data["prices"],
        data["per_pbr"],
        data["flows"]
    )
    
    print(f"리스크 스코어 계산 완료")
    print(f"현재 리스크 스코어: {risk_scores.iloc[-1]:.2f}")
    print(f"데이터 기간: {risk_scores.index[0].strftime('%Y-%m-%d')} ~ {risk_scores.index[-1].strftime('%Y-%m-%d')}")

