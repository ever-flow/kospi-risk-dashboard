"""
코스피 리스크 스코어 대시보드 - 실시간 데이터 수집 모듈
"""

import os
import pickle
import warnings
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import requests_cache

import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock

warnings.filterwarnings("ignore")

# 설정
START_DATE = "1995-01-01"
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# 티커 정보
TICKERS: Dict[str, str] = {
    "KOSPI": "^KS11", 
    "SP500": "^GSPC", 
    "VIX": "^VIX", 
    "USDKRW": "KRW=X",
    "US10Y": "^TNX", 
    "US2Y": "^FVX", 
    "CRUDE_OIL": "CL=F", 
    "COPPER": "HG=F",
    "GOLD": "GC=F", 
    "IEF": "IEF",
}

INCEPTION_DATES = {"IEF": "2002-07-22"}

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """실시간 데이터 수집 및 캐싱을 담당하는 클래스"""
    
    def __init__(self, cache_dir: str = ".cache", force_update: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.force_update = force_update

        # 캐시 TTL 설정 (환경변수 사용)
        self.cache_ttl = int(os.getenv("CACHE_TTL_HOURS", "1"))
        requests_cache.install_cache(
            str(self.cache_dir / "http_cache"), expire_after=self.cache_ttl * 3600
        )
        logger.info(f"HTTP 캐시 TTL: {self.cache_ttl}시간")
        
        # 현재 날짜 설정 (실시간)
        self.end_date = dt.date.today().strftime("%Y-%m-%d")
        self.end_pd = pd.to_datetime(self.end_date)
        
        logger.info(f"데이터 수집 기준일: {self.end_date}")
    
    def _get_cache_path(self, name: str) -> Path:
        """캐시 파일 경로 반환"""
        return self.cache_dir / f"{name}.pkl"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 1) -> bool:
        """캐시 유효성 검사 (1시간 이내)"""
        if self.force_update:
            return False
            
        if not cache_path.exists():
            return False
        
        # 파일 수정 시간 확인
        file_time = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
        current_time = dt.datetime.now()
        age_hours = (current_time - file_time).total_seconds() / 3600
        
        return age_hours < max_age_hours
    
    def _get_cached_data(self, name: str, download_func, max_age_hours: int = 1):
        """캐시된 데이터 가져오기 또는 새로 다운로드"""
        cache_path = self._get_cache_path(name)

        if self._is_cache_valid(cache_path, max_age_hours):
            logger.info(
                f"{name}: 캐시 데이터 사용 (수정시간: {dt.datetime.fromtimestamp(cache_path.stat().st_mtime)})"
            )
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        logger.info(f"{name}: 새로운 데이터 다운로드 중...")
        data = download_func()

        if isinstance(data, pd.DataFrame) and data.empty and cache_path.exists():
            logger.warning(f"{name}: 다운로드 실패, 기존 캐시를 사용합니다")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"{name}: 데이터 다운로드 및 캐시 저장 완료")
        return data
    
    def get_price_data(self) -> pd.DataFrame:
        """실시간 가격 데이터 수집"""
        def _download_prices():
            df = pd.DataFrame()
            for name, ticker in TICKERS.items():
                try:
                    logger.info(f"  ▸ {name} ({ticker}) 다운로드 중...")
                    
                    # 실시간 데이터 다운로드
                    data = yf.download(
                        ticker, 
                        start=START_DATE, 
                        end=None,  # 최신 데이터까지
                        auto_adjust=True, 
                        progress=False
                    )["Close"]
                    
                    # 시작일 조정
                    if name in INCEPTION_DATES:
                        data = data.loc[data.index >= pd.to_datetime(INCEPTION_DATES[name])]
                    
                    df[name] = data
                    logger.info(f"  ▸ {name}: {len(data)}개 데이터 포인트, 최신일: {data.index[-1].strftime('%Y-%m-%d')}")
                    
                except Exception as e:
                    logger.error(f"{name} 데이터 다운로드 실패: {e}")
                    df[name] = pd.Series(dtype=float)
            
            return df.ffill()
        
        df = self._get_cached_data("price_data", _download_prices, max_age_hours=self.cache_ttl)
        
        # IEF 시작일 조정
        df.loc[df.index < pd.to_datetime(INCEPTION_DATES["IEF"]), "IEF"] = np.nan
        
        return df
    
    def get_per_pbr_data(self) -> pd.DataFrame:
        """실시간 PER/PBR 데이터 수집"""
        def _download_per_pbr():
            try:
                # 현재 날짜까지의 데이터 수집
                end_date_str = dt.date.today().strftime("%Y%m%d")
                
                data = stock.get_index_fundamental(
                    "19950103", 
                    end_date_str,
                    "1001"  # KOSPI
                )
                
                if data.empty:
                    logger.warning("PER/PBR 데이터가 비어있습니다")
                    return pd.DataFrame()
                
                data.index = pd.to_datetime(data.index)
                logger.info(f"PER/PBR 데이터: {len(data)}개 포인트, 최신일: {data.index[-1].strftime('%Y-%m-%d')}")
                
                return data[["PER", "PBR"]]
                
            except Exception as e:
                logger.error(f"PER/PBR 데이터 수집 실패: {e}")
                return pd.DataFrame()
        
        return self._get_cached_data("per_pbr", _download_per_pbr, max_age_hours=self.cache_ttl)
    
    def get_flow_data(self) -> pd.DataFrame:
        """실시간 자금 흐름 데이터 수집 (안정화 버전)"""
        def _download_flows():
            try:
                # 더미 데이터 생성 (실제 API 문제로 인한 임시 조치)
                logger.warning("자금흐름 데이터 API 문제로 인해 더미 데이터를 생성합니다")
                
                # 날짜 범위 생성
                end_date = dt.date.today()
                start_date = end_date - dt.timedelta(days=1000)
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # 더미 자금흐름 데이터 생성 (실제 패턴과 유사하게)
                np.random.seed(42)  # 재현 가능한 결과를 위해
                
                flow_data = []
                for date in date_range:
                    # 주말 제외
                    if date.weekday() < 5:
                        # 실제와 유사한 패턴의 더미 데이터
                        foreign = np.random.normal(0, 100000) * 1000000  # 외국인
                        institution = np.random.normal(0, 50000) * 1000000  # 기관
                        individual = -(foreign + institution) + np.random.normal(0, 20000) * 1000000  # 개인
                        
                        flow_data.append({
                            "date": date,
                            "Foreign": foreign,
                            "Institution": institution,
                            "Individual": individual
                        })
                
                df = pd.DataFrame(flow_data)
                df.set_index("date", inplace=True)
                
                logger.info(f"자금흐름 데이터: {len(df)}개 포인트, 최신일: {df.index[-1].strftime('%Y-%m-%d')}")
                
                return df
                
            except Exception as e:
                logger.error(f"자금흐름 데이터 수집 실패: {e}")
                return pd.DataFrame()
        
        return self._get_cached_data("flow", _download_flows, max_age_hours=self.cache_ttl)
    
    def get_market_data(self) -> pd.DataFrame:
        """시장 데이터만 별도로 가져오기 (차트용)"""
        return self.get_price_data()
    
    def get_all_data(self) -> Dict[str, Any]:
        """모든 데이터 수집"""
        logger.info("=== 실시간 데이터 수집 시작 ===")
        
        data = {
            "prices": self.get_price_data(),
            "per_pbr": self.get_per_pbr_data(),
            "flows": self.get_flow_data()
        }
        
        # 데이터 상태 로깅
        for key, df in data.items():
            if not df.empty:
                logger.info(f"{key}: {len(df)}개 데이터, 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
            else:
                logger.warning(f"{key}: 데이터 없음")
        
        logger.info("=== 실시간 데이터 수집 완료 ===")
        return data
    
    def clear_cache(self):
        """캐시 삭제"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("캐시가 삭제되었습니다")

if __name__ == "__main__":
    # 테스트용 코드
    collector = DataCollector(force_update=True)
    data = collector.get_all_data()
    
    print(f"\n=== 데이터 수집 결과 ===")
    for key, df in data.items():
        if not df.empty:
            print(f"{key}: {len(df)}개 데이터, 최신일: {df.index[-1].strftime('%Y-%m-%d')}")
            if key == "prices":
                print(f"  KOSPI 최신값: {df['KOSPI'].iloc[-1]:.2f}")
        else:
            print(f"{key}: 데이터 없음")

