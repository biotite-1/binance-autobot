import pandas as pd
from binance.client import Client
import ta  # 기술적 지표 라이브러리

api_key = "YE1oDI3c5sKXzzzsO6KFgPZZUv85hzKyd9oPJJTGFLR7scfau5fhwWujuTujIStG"
api_secret = "B1fWntUUh3dHUl0vl9hkrtp0byxCLmhX7CcPoQJNz9JUoo1171BQMjikvxcUGLgS"
client = Client(api_key, api_secret, testnet=True)

# 최근 100개 1시간 봉 데이터 가져오기
klines = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, limit=100)

# 데이터프레임 변환
df = pd.DataFrame(klines, columns=['time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
df['close'] = df['close'].astype(float)

# RSI 계산
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

# 최근 RSI 확인
latest_rsi = df['rsi'].iloc[-1]
print(f"현재 RSI: {latest_rsi:.2f}")

# 매매 조건
if latest_rsi < 30:
    print("🚀 매수 신호 (과매도 구간)")
elif latest_rsi > 70:
    print("💀 매도 신호 (과매수 구간)")
else:
    print("⏸ 대기 상태")


