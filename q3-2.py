import pandas as pd
from binance.client import Client
import ta
import time
from datetime import datetime

# ✅ Binance Testnet 연결
api_key = "YE1oDI3c5sKXzzzsO6KFgPZZUv85hzKyd9oPJJTGFLR7scfau5fhwWujuTujIStG"
api_secret = "B1fWntUUh3dHUl0vl9hkrtp0byxCLmhX7CcPoQJNz9JUoo1171BQMjikvxcUGLgS"
client = Client(api_key, api_secret, testnet=True)

symbol = "BTCUSDT"
quantity = 0.001

# ✅ 거래 로그 파일 초기화
log_file = "trade_log.csv"
try:
    df_log = pd.read_csv(log_file)
except FileNotFoundError:
    df_log = pd.DataFrame(columns=["datetime", "symbol", "side", "price", "quantity", "rsi"])
    df_log.to_csv(log_file, index=False)

# ✅ RSI 계산 함수
def get_rsi(symbol):
    klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=100)
    df = pd.DataFrame(klines, columns=[
        'time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'
    ])
    df['close'] = df['close'].astype(float)
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch'] = stoch.stoch()

    return df['stoch'].iloc[-1], df['close'].iloc[-1]

# ✅ 매매 실행 + 기록 함수
def execute_trade(side, price, rsi):
    order = None
    if side == "BUY":
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
    elif side == "SELL":
        order = client.order_market_sell(symbol=symbol, quantity=quantity)

    log_entry = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "side": side,
        "price": price,
        "quantity": quantity,
        "rsi": rsi
    }

    # 로그 파일에 저장
    global df_log
    df_log = pd.concat([df_log, pd.DataFrame([log_entry])])
    df_log.to_csv(log_file, index=False)
    print(f"💾 거래 기록 저장됨: {log_entry}")

# ✅ 메인 루프
while True:
    try:
        rsi, price = get_rsi(symbol)
        print(f"\n현재가: {price:.2f} | RSI: {rsi:.2f}")

        if rsi < 20:
            print("🚀 매수 신호 발생 → 주문 실행")
            execute_trade("BUY", price, rsi)
        elif rsi > 80:
            print("💀 매도 신호 발생 → 주문 실행")
            execute_trade("SELL", price, rsi)
        else:
            print("⏸ 대기 중...")

        time.sleep(60)

    except Exception as e:
        print("❌ 오류 발생:", e)
        time.sleep(60)
