#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC/USDT 자동 매매 봇 (AI 신호 + 페이퍼/실거래 공용)

⚠️ 중요한 고지
- 본 코드는 교육/연구용 예시입니다. 수익을 보장하지 않습니다. 실제 자금 투입 전 반드시 페이퍼(trading_mode=paper)로 충분히 테스트하세요.
- 거래는 각 지역의 법/규제를 준수하는 합법적인 거래소/계정에서만 하세요. 접근 제한 우회, 약관 위반 등을 절대 시도하지 마세요.
- 실거래는 본인 책임입니다. 손실이 발생할 수 있습니다.

핵심 기능
- ccxt를 사용해 캔들 수집 (기본: Binance, 심볼: BTC/USDT, 타임프레임: 1m)
- 간단한 온라인 로지스틱 회귀(Perceptron 유사)로 신호 생성 (특징량: 수익률, EMA 스프레드, RSI, 변동성 등)
- 포지션: 롱(현물 기준)만 진입/청산 (기본은 페이퍼 트레이딩)
- 각 트레이드 청산 시 수익률(%)을 콘솔/CSV로 출력
- 기본 수수료/슬리피지 파라미터 반영

빠른 시작
1) 필수 설치
   pip install ccxt pandas numpy

2) 실행 (페이퍼 트레이딩, 기본값)
   python btcusdt_ai_trader.py

3) (선택) 실거래 전환 — 합법적 계정/지역에서만!
   환경변수 설정 후 실행하세요.
   - EXCHANGE=binance  (또는 지원 거래소명, 예: bybit, okx 등 — ccxt 지원 목록 참조)
   - API_KEY=...  API_SECRET=...
   - LIVE=1  (1이면 실거래 모드)
   예)
   LIVE=1 EXCHANGE=binance API_KEY=xxxx API_SECRET=yyyy python btcusdt_ai_trader.py

4) 로그 확인
   - trades.csv : 각 거래 기록 (진입/청산/수익률)
   - 콘솔 : 트레이드 종료 시 수익률을 바로 출력

설정값(환경변수)
- EXCHANGE (기본 binance)
- SYMBOL (기본 BTC/USDT)
- TIMEFRAME (기본 1m)
- LIVE (기본 0: 페이퍼, 1: 실거래)
- POSITION_SIZE_PCT (기본 0.2 = 가용 현금의 20%)
- FEE_RATE (기본 0.001 = 0.1% 한쪽)
- TP_PCT, SL_PCT (기본 0.01, 0.01 = ±1%에서 익절/손절)
- BUY_TH, SELL_TH (기본 0.55, 0.45 = 진입/청산 신호 임계값)
- MIN_BALANCE_USDT (기본 20)

참고
- 심볼 표기는 ccxt 기준 "BTC/USDT" 형태입니다.
- 일부 거래소/지역에서는 API 접근이 제한될 수 있습니다(공개 시세/거래 모두). 해당 시엔 다른 합법적 거래소 또는 데이터 소스를 사용하세요.
"""

import os
import sys
import time
import math
import json
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    print("[ERROR] ccxt가 설치되어 있지 않습니다. 먼저 'pip install ccxt'를 실행하세요.")
    sys.exit(1)

# ===================== 설정 =====================
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
LIVE = int(os.getenv("LIVE", "0")) == 1
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.2"))   # 가용 현금의 20%
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))                   # 0.1%/사이드
TP_PCT = float(os.getenv("TP_PCT", "0.01"))                         # +1% 익절
SL_PCT = float(os.getenv("SL_PCT", "0.01"))                         # -1% 손절
BUY_TH = float(os.getenv("BUY_TH", "0.55"))
SELL_TH = float(os.getenv("SELL_TH", "0.45"))
MIN_BALANCE_USDT = float(os.getenv("MIN_BALANCE_USDT", "20"))
TRAIN_WINDOW = int(os.getenv("TRAIN_WINDOW", "400"))                # 학습 샘플 수
SLEEP_SEC = int(os.getenv("SLEEP_SEC", "10"))                        # 루프 대기
TRADES_CSV = os.getenv("TRADES_CSV", "trades.csv")

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.width', 180)

# ===================== 유틸: 지표 =====================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index).rolling(period).mean()
    loss = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ===================== 온라인 로지스틱 회귀 =====================
class OnlineLogReg:
    def __init__(self, n_features: int, lr: float = 0.05, l2: float = 1e-4):
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    def _sigmoid(self, z):
        # 안정적 시그모이드
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    def predict_proba(self, x: np.ndarray) -> float:
        return float(self._sigmoid(np.dot(self.w, x) + self.b))

    def update(self, x: np.ndarray, y: float):
        # y in {0,1}
        p = self.predict_proba(x)
        # 손실: 로지스틱(크로스엔트로피), 그라디언트
        grad_w = (p - y) * x + self.l2 * self.w
        grad_b = (p - y)
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b

# ===================== 데이터/특징 생성 =====================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame -> 특징량 DataFrame
    df: index=ms timestamp, columns=[timestamp, open, high, low, close, volume]
    """
    close = df['close']
    ret1 = close.pct_change().fillna(0.0)
    ret3 = close.pct_change(3)
    ret5 = close.pct_change(5)

    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema_spread = (ema9 - ema21) / (close + 1e-12)

    rsi14 = rsi(close, 14) / 100.0  # 0~1로 정규화

    vol = ret1.rolling(20).std().fillna(0.0)

    feats = pd.DataFrame({
        'ret1': ret1,
        'ret3': ret3,
        'ret5': ret5,
        'ema_spread': ema_spread,
        'rsi14': rsi14,
        'vol20': vol,
    })

    # 결측 제거
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats

def build_labels(df: pd.DataFrame, horizon: int = 3) -> pd.Series:
    close = df['close']
    future_ret = close.shift(-horizon).pct_change(periods=horizon)
    # 미래 3분 수익률 > 0 => 1, else 0
    y = (future_ret > 0).astype(int)
    y = y.loc[df.index]
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    return y

# ===================== 브로커 래퍼 =====================
class Broker:
    def __init__(self, exchange_name: str, live: bool):
        self.exchange_name = exchange_name
        self.live = live
        params = {
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            }
        }
        if live and API_KEY and API_SECRET:
            params['apiKey'] = API_KEY
            params['secret'] = API_SECRET
        self.ex = getattr(ccxt, exchange_name)(params)
        try:
            self.ex.load_markets()
        except Exception as e:
            print(f"[WARN] 시장 로드 실패: {e}")

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        raw = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df

    def fetch_price(self, symbol: str) -> float:
        ticker = self.ex.fetch_ticker(symbol)
        return float(ticker['last'])

    def fetch_usdt_balance(self) -> float:
        if not self.live:
            raise RuntimeError("페이퍼 모드에서는 실제 잔고를 사용할 수 없습니다.")
        balances = self.ex.fetch_balance()
        # USDT 또는 USD 계정
        for code in ["USDT", "USD"]:
            if code in balances['total']:
                return float(balances['free'].get(code, 0.0))
        return 0.0

    def market_buy(self, symbol: str, amount: float):
        if not self.live:
            return {"status": "paper", "side": "buy", "amount": amount}
        return self.ex.create_market_buy_order(symbol, amount)

    def market_sell(self, symbol: str, amount: float):
        if not self.live:
            return {"status": "paper", "side": "sell", "amount": amount}
        return self.ex.create_market_sell_order(symbol, amount)

# ===================== 트레이딩 엔진 =====================
class Trader:
    def __init__(self, broker: Broker):
        self.broker = broker
        self.model = None
        self.in_position = False
        self.entry_price = None
        self.qty = 0.0
        self.cash = 1000.0 if not broker.live else None  # 페이퍼 시작 현금
        self.asset = 0.0
        self.cum_return = 1.0
        # CSV 헤더 생성
        if not os.path.exists(TRADES_CSV):
            pd.DataFrame(columns=['entry_time','entry_price','exit_time','exit_price','qty','pnl_pct','cum_return']).to_csv(TRADES_CSV, index=False)

    def train_model(self, feats: pd.DataFrame, labels: pd.Series):
        # 공통 인덱스
        idx = feats.index.intersection(labels.index)
        X = feats.loc[idx].values
        y = labels.loc[idx].values
        if len(X) < TRAIN_WINDOW + 1:
            return False
        X = X[-TRAIN_WINDOW:]
        y = y[-TRAIN_WINDOW:]
        # 표준화
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sd

        n_feat = Xn.shape[1]
        self.model = {
            'clf': OnlineLogReg(n_feat, lr=0.05, l2=1e-4),
            'mu': mu,
            'sd': sd,
            'last_feat': None
        }
        clf = self.model['clf']
        # 온라인 업데이트
        for i in range(len(Xn)):
            clf.update(Xn[i], float(y[i]))
        return True

    def predict_proba(self, x_row: np.ndarray) -> float:
        mu = self.model['mu']; sd = self.model['sd']; clf = self.model['clf']
        xn = (x_row - mu) / sd
        p = clf.predict_proba(xn)
        return p

    def paper_portfolio_value(self, price: float) -> float:
        return self.cash + self.asset * price

    def log_trade(self, entry_time, entry_price, exit_time, exit_price, qty, pnl_pct):
        self.cum_return *= (1.0 + pnl_pct)
        row = {
            'entry_time': entry_time.isoformat(),
            'entry_price': entry_price,
            'exit_time': exit_time.isoformat(),
            'exit_price': exit_price,
            'qty': qty,
            'pnl_pct': pnl_pct,
            'cum_return': self.cum_return,
        }
        pd.DataFrame([row]).to_csv(TRADES_CSV, mode='a', header=False, index=False)
        # 요구사항: 거래할 때마다 수익률 출력
        sign = "수익" if pnl_pct >= 0 else "손실"
        print(f"[TRADE] {sign} {pnl_pct*100:.2f}% | 누적배율 {self.cum_return:.4f} | 진입 {entry_time}, 청산 {exit_time}")

    def maybe_trade(self):
        try:
            df = self.broker.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=max(1000, TRAIN_WINDOW + 50))
        except Exception as e:
            print(f"[ERROR] OHLCV 수집 실패: {e}")
            return

        feats = build_features(df)
        labels = build_labels(df)
        if not self.model:
            ok = self.train_model(feats, labels)
            if not ok:
                print("[INFO] 데이터가 더 필요합니다. 대기 중...")
                return

        # 최신 특징
        common_idx = feats.index.intersection(labels.index)
        if len(common_idx) == 0:
            return
        last_ts = common_idx[-1]
        x_row = feats.loc[last_ts].values
        p_up = self.predict_proba(x_row)
        last_price = float(df.loc[last_ts, 'close'])

        # 포지션 관리 (롱만)
        global BUY_TH, SELL_TH
        if not self.in_position:
            # 진입 조건
            if p_up >= BUY_TH:
                # 수량 계산
                if self.broker.live:
                    try:
                        usdt = self.broker.fetch_usdt_balance()
                        alloc = max(0.0, usdt * POSITION_SIZE_PCT)
                        if alloc < MIN_BALANCE_USDT:
                            print(f"[WARN] 잔고 부족 또는 MIN_BALANCE_USDT({MIN_BALANCE_USDT}) 미만. 건너뜀")
                            return
                        qty = (alloc * (1 - FEE_RATE)) / last_price
                        order = self.broker.market_buy(SYMBOL, qty)
                        self.qty = qty
                        self.entry_price = last_price * (1 + FEE_RATE)  # 슬리피지/수수료 고려
                        self.in_position = True
                        print(f"[LIVE] 매수 진입 qty={qty:.6f} @~{self.entry_price:.2f}, p_up={p_up:.3f}")
                    except Exception as e:
                        print(f"[ERROR] 매수 실패: {e}")
                        return
                else:
                    # 페이퍼
                    alloc = max(0.0, self.cash * POSITION_SIZE_PCT)
                    if alloc < MIN_BALANCE_USDT:
                        # 초기 현금이 적거나 설정이 커서 못 살 수 있음
                        alloc = self.cash  # 가능한 전액
                    qty = (alloc * (1 - FEE_RATE)) / last_price
                    cost = qty * last_price * (1 + FEE_RATE)
                    if cost > self.cash:
                        print(f"[PAPER] 현금 부족으로 진입 실패 (필요 {cost:.2f} > 보유 {self.cash:.2f})")
                        return
                    self.cash -= cost
                    self.asset += qty
                    self.qty = qty
                    self.entry_price = last_price * (1 + FEE_RATE)
                    self.in_position = True
                    print(f"[PAPER] 매수 진입 qty={qty:.6f} @~{self.entry_price:.2f}, p_up={p_up:.3f}")
        else:
            # 청산 조건: 확률 하락, TP/SL 도달
            take_profit = (last_price - self.entry_price) / self.entry_price >= TP_PCT
            stop_loss = (last_price - self.entry_price) / self.entry_price <= -SL_PCT
            exit_signal = (p_up <= SELL_TH) or take_profit or stop_loss

            if exit_signal:
                if self.broker.live:
                    try:
                        order = self.broker.market_sell(SYMBOL, self.qty)
                        exit_price = last_price * (1 - FEE_RATE)
                        pnl_pct = (exit_price - self.entry_price) / self.entry_price
                        self.log_trade(entry_time=last_ts, entry_price=self.entry_price,
                                       exit_time=last_ts, exit_price=exit_price,
                                       qty=self.qty, pnl_pct=pnl_pct)
                        self.in_position = False
                        self.qty = 0.0
                        self.entry_price = None
                        print(f"[LIVE] 매도 청산 @~{exit_price:.2f} | TP:{take_profit} SL:{stop_loss} p_up={p_up:.3f}")
                    except Exception as e:
                        print(f"[ERROR] 매도 실패: {e}")
                        return
                else:
                    # 페이퍼
                    exit_price = last_price * (1 - FEE_RATE)
                    proceeds = self.qty * exit_price
                    self.asset -= self.qty
                    self.cash += proceeds
                    pnl_pct = (exit_price - self.entry_price) / self.entry_price
                    self.log_trade(entry_time=last_ts, entry_price=self.entry_price,
                                   exit_time=last_ts, exit_price=exit_price,
                                   qty=self.qty, pnl_pct=pnl_pct)
                    print(f"[PAPER] 매도 청산 @~{exit_price:.2f} | TP:{take_profit} SL:{stop_loss} p_up={p_up:.3f}")
                    self.qty = 0.0
                    self.entry_price = None
                    self.in_position = False

    def run(self):
        print(f"=== 시작 | LIVE={self.broker.live} | {EXCHANGE_NAME} {SYMBOL} {TIMEFRAME} ===")
        while True:
            try:
                self.maybe_trade()
            except Exception as e:
                print("[FATAL] 루프 오류:", e)
                traceback.print_exc()
            time.sleep(SLEEP_SEC)


def main():
    print("[INFO] 규제/약관을 준수하는 합법적 계정/지역에서만 실거래(LIVE=1)를 사용하세요.")
    if LIVE and (not API_KEY or not API_SECRET):
        print("[ERROR] LIVE=1 모드에는 API_KEY, API_SECRET 환경변수가 필요합니다.")
        sys.exit(1)

    broker = Broker(EXCHANGE_NAME, live=LIVE)
    trader = Trader(broker)
    trader.run()


if __name__ == "__main__":
    main()



