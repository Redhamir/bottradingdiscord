"""
bot_yahoo_full.py
Bot Discord 100% Yahoo Finance -- signaux techniques multi-timeframe.

Utilisation:
- Définir variables d'environnement DISCORD_TOKEN et DISCORD_CHANNEL (nom du channel ou ID)
- Lancer: python bot_yahoo_full.py
- Hébergement recommandé: Replit / Render / Railway (voir README)
"""

import os
import asyncio
import math
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
import discord
from discord import Embed
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ========== CONFIG ==========
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL = os.getenv("DISCORD_CHANNEL", "trading-signals")  # name or ID
# Symbols (Yahoo tickers)
SYMBOLS = [
    "BTC-USD","ETH-USD","XLM-USD","ADA-USD","XTZ-USD","LTC-USD","DOT-USD","LINK-USD",
    "TSLA","AAPL","ORCL","V","NVDA","AMD","GOOGL"
]

# Timeframes to monitor and mapping to minutes (1S interpreted as 1w)
TF_TO_MIN = {
    "1m": 1, "5m": 5, "10m": 10, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440, "3d": 4320, "1w": 10080
}
# Map timeframe to trading style
TF_TO_STYLE = {
    "1m":"scalping","5m":"scalping","10m":"scalping","15m":"intraday","30m":"intraday",
    "1h":"intraday","2h":"intraday","4h":"swing","1d":"swing","3d":"swing","1w":"swing"
}

# Alert cooldown (seconds) per symbol+tf+signal to avoid spam
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "60"))

# How many historical minutes to pull (1m base) -> we'll use 7 days by default
BASE_PERIOD = os.getenv("BASE_PERIOD", "7d")  # yfinance supported period like "7d", "30d"
# Resampling rule if needed (we fetch 1m then aggregate)
MAX_BARS = 1000  # cap bars to keep memory reasonable

# Safety
if not DISCORD_TOKEN:
    raise Exception("Define DISCORD_TOKEN env var")

# ========== Discord client ==========
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Simple in-memory dedupe store { key: last_timestamp }
last_alert_ts = {}

# ========== Utilities & Indicators ==========
def fmt_price(p):
    try:
        return f"{float(p):.2f}"
    except:
        return str(p)

def now_str():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_vwap(df: pd.DataFrame, window:int=60) -> pd.Series:
    pv = (df['close'] * df['volume']).rolling(window=window, min_periods=1).sum()
    v = df['volume'].rolling(window=window, min_periods=1).sum()
    return (pv / v).fillna(method='bfill')

def detect_candlestick_patterns(df: pd.DataFrame):
    """
    Very simple detectors for hammer, shooting star, engulfing
    Returns list of strings
    """
    patterns = []
    if len(df) < 3:
        return patterns
    o = df['open'].iloc[-1]; c = df['close'].iloc[-1]; h = df['high'].iloc[-1]; l = df['low'].iloc[-1]
    body = abs(c - o)
    candle_range = h - l if h-l>0 else 1e-9
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    # Hammer: small body at top, long lower wick
    if lower_wick > 2*body and body / candle_range < 0.3:
        patterns.append("Hammer (possible reversal)")
    # Shooting star: long upper wick
    if upper_wick > 2*body and body / candle_range < 0.3:
        patterns.append("Shooting star (possible reversal)")
    # Bullish engulfing (requires previous candle)
    prev_o = df['open'].iloc[-2]; prev_c = df['close'].iloc[-2]
    if prev_c < prev_o and c > o and c > prev_o and o < prev_c:
        patterns.append("Bullish Engulfing")
    if prev_c > prev_o and c < o and c < prev_o and o > prev_c:
        patterns.append("Bearish Engulfing")
    return patterns

# classify signal -> trading types
def classify_signal(signal_code:str, tf:str):
    base = TF_TO_STYLE.get(tf, "intraday")
    # heuristics
    if "spike" in signal_code.lower() or tf in ["1m","5m","10m"]:
        return ["scalping","intraday"]
    if "breakout" in signal_code.lower():
        return [base]
    if "rsi" in signal_code.lower():
        return ["intraday","swing"]
    return [base]

# heuristic confidence
def confidence_score(signal_code:str, tf:str, df:pd.DataFrame):
    score = 0.5
    if "volume" in signal_code.lower(): score += 0.15
    if "breakout" in signal_code.lower(): score += 0.15
    if tf in ["1d","3d","1w"]: score += 0.08
    # increase confidence if multiple confirmations
    if len(df)>0:
        close = df['close']
        ma9 = sma(close,9).iloc[-1] if len(close)>9 else None
        ma21 = sma(close,21).iloc[-1] if len(close)>21 else None
        if ma9 and ma21 and ma9>ma21: score += 0.05
    return min(0.95, score)

# ========== FETCHING & RESAMPLING ==========
async def fetch_yahoo_1m(symbol:str, period:str=BASE_PERIOD):
    """
    Fetch 1m bars for period then return pandas DataFrame.
    We use yfinance.download with interval=1m.
    """
    try:
        df = yf.download(tickers=symbol, period=period, interval="1m", progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        df = df[['open','high','low','close','volume']].astype(float)
        # keep last MAX_BARS
        if len(df) > MAX_BARS:
            df = df.iloc[-MAX_BARS:]
        return df
    except Exception as e:
        print(f"[fetch error] {symbol} : {e}")
        return None

def resample_df(df_1m:pd.DataFrame, tf_min:int) -> pd.DataFrame:
    """
    Resample 1m dataframe to desired tf in minutes.
    tf_min in minutes. For weekly/day, use rules.
    """
    if tf_min == 1:
        return df_1m.copy()
    rule = f"{tf_min}T"
    df = df_1m.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    df.dropna(inplace=True)
    return df

# ========== SIGNAL ENGINE ==========
def detect_signals(df:pd.DataFrame):
    """
    Return list of tuples (code, human_text)
    """
    sigs = []
    if df is None or len(df) < 20:
        return sigs

    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']

    # RSI
    rsi_series = compute_rsi(close)
    rsi_val = rsi_series.iloc[-1]
    if rsi_val > 70:
        sigs.append(("RSI_over_70", "RSI > 70 (surachat)"))
    elif rsi_val < 30:
        sigs.append(("RSI_under_30", "RSI < 30 (survente)"))

    # MA cross
    ma9 = sma(close, 9)
    ma21 = sma(close,21)
    if len(ma21.dropna())>1:
        if ma9.iloc[-2] < ma21.iloc[-2] and ma9.iloc[-1] > ma21.iloc[-1]:
            sigs.append(("MA9_cross_MA21_up", "MA9 a croisé au-dessus de MA21"))
        if ma9.iloc[-2] > ma21.iloc[-2] and ma9.iloc[-1] < ma21.iloc[-1]:
            sigs.append(("MA9_cross_MA21_down", "MA9 a croisé en-dessous de MA21"))

    # Breakout high (20)
    if len(high) >= 21:
        prev_high20 = high.rolling(20).max().shift(1)
        if close.iloc[-1] > prev_high20.iloc[-1]:
            sigs.append(("Breakout_20", "Cassure des 20 derniers hauts"))

    # VWAP cross (rolling 60)
    vw = compute_vwap(df, window=min(60, max(10,int(len(df)/5))))
    if len(vw)>2:
        if df['close'].iloc[-2] < vw.iloc[-2] and df['close'].iloc[-1] > vw.iloc[-1]:
            sigs.append(("VWAP_cross_up", "Prix a croisé au dessus du VWAP (session)"))
        if df['close'].iloc[-2] > vw.iloc[-2] and df['close'].iloc[-1] < vw.iloc[-1]:
            sigs.append(("VWAP_cross_down", "Prix a croisé en dessous du VWAP (session)"))

    # Volume spike
    vol_ma20 = vol.rolling(20).mean().fillna(method='bfill')
    if vol_ma20.iloc[-1] > 0 and vol.iloc[-1] > 3 * vol_ma20.iloc[-1]:
        if close.iloc[-1] > close.iloc[-2]:
            sigs.append(("Volume_spike_buy", "Spike volume + prix up (achat agressif)"))
        else:
            sigs.append(("Volume_spike_sell", "Spike volume + prix down (vente agressive)"))

    # RSI divergence (simple heuristic)
    # bullish divergence: price makes lower low while rsi makes higher low
    try:
        # pick two recent local lows
        window = min(60, len(close))
        recent = df[-window:]
        # get two lowest indices
        lows = recent['low'].nsmallest(4).index
        if len(lows) >= 2:
            i1, i2 = lows[-1], lows[-2]
            if recent.loc[i2,'low'] < recent.loc[i1,'low'] and compute_rsi(recent['close']).loc[i2] > compute_rsi(recent['close']).loc[i1]:
                sigs.append(("RSI_div_bull", "Divergence RSI haussière (approx.)"))
        highs = recent['high'].nlargest(4).index
        if len(highs) >= 2:
            j1, j2 = highs[-1], highs[-2]
            if recent.loc[j2,'high'] > recent.loc[j1,'high'] and compute_rsi(recent['close']).loc[j2] < compute_rsi(recent['close']).loc[j1]:
                sigs.append(("RSI_div_bear", "Divergence RSI baissière (approx.)"))
    except Exception:
        pass

    # candle patterns
    patterns = detect_candlestick_patterns(df)
    for p in patterns:
        sigs.append(("CANDLE_"+p.replace(" ","_"), p))

    # add more heuristics if needed...
    return sigs

# ========== MESSAGE BUILD ==========
def decide_action_from_signal(code:str):
    code_low = code.lower()
    if any(k in code_low for k in ["buy","bull","under_30","cross_up"]):
        return "BUY"
    if any(k in code_low for k in ["sell","over_70","cross_down","down","bear"]):
        return "SELL"
    # default: if contains 'buy' words choose buy, else sell if negative words
    if "buy" in code_low or "up" in code_low:
        return "BUY"
    return "SELL"

def build_embed(symbol:str, price:float, tf:str, signal_code:str, signal_text:str, conf:float, types:list):
    action = decide_action_from_signal(signal_code)
    styles = ", ".join(types)
    e = Embed(title=f"{symbol}  —  {fmt_price(price)}", color=0x1abc9c if action=="BUY" else 0xe74c3c)
    e.add_field(name="Action", value=action, inline=True)
    e.add_field(name="Signal", value=signal_text, inline=True)
    e.add_field(name="Timeframe", value=tf, inline=True)
    e.add_field(name="Type de trading", value=styles, inline=True)
    e.add_field(name="Confiance", value=f"{conf:.2f}", inline=True)
    e.set_footer(text=f"{now_str()} • Powered by Yahoo Finance (no API key)")
    return e

# ========== MAIN MONITOR LOOP ==========
async def monitor_loop(channel):
    """
    Master loop:
    - For each symbol, fetch 1m data (yfinance)
    - For each timeframe, resample and detect signals
    - Send embed if new
    """
    # We fetch 1m once per symbol, then resample locally for all TFs
    while True:
        try:
            for symbol in SYMBOLS:
                # fetch 1m base
                df1m = await fetch_yahoo_1m(symbol, period=BASE_PERIOD)
                if df1m is None or df1m.empty:
                    print(f"[warning] pas de données pour {symbol}")
                    await asyncio.sleep(1)
                    continue
                # ensure datetime index is timezone-aware
                if not isinstance(df1m.index, pd.DatetimeIndex):
                    df1m.index = pd.to_datetime(df1m.index)
                # iterate timeframes
                for tf, minutes in TF_TO_MIN.items():
                    df_tf = resample_df(df1m, minutes)
                    if df_tf is None or len(df_tf) < 20:
                        continue
                    signals = detect_signals(df_tf)
                    price = df_tf['close'].iloc[-1]
                    for code, text in signals:
                        key = f"{symbol}|{tf}|{code}"
                        ts = asyncio.get_event_loop().time()
                        last = last_alert_ts.get(key, 0)
                        if ts - last < ALERT_COOLDOWN:
                            continue  # within cooldown
                        types = classify_signal(code, tf)
                        conf = confidence_score(code, tf, df_tf)
                        embed = build_embed(symbol, price, tf, code, text, conf, types)
                        try:
                            await channel.send(embed=embed)
                        except Exception as e:
                            # fallback simple text
                            try:
                                await channel.send(f"{symbol} {fmt_price(price)} {decide_action_from_signal(code)} — {text} — TF {tf} — type {', '.join(types)} — conf {conf:.2f}")
                            except Exception as ex:
                                print("Discord send failed:", ex)
                        last_alert_ts[key] = ts
                        # small pause to avoid burst
                        await asyncio.sleep(0.5)
                # small pause between symbols
                await asyncio.sleep(1.0)
            # after one full round, wait a short time before next round
            await asyncio.sleep(5)
        except Exception as e:
            print("Monitor loop error:", e)
            traceback.print_exc()
            await asyncio.sleep(5)

# ========== Discord event handlers ==========
@client.event
async def on_ready():
    print("Bot ready:", client.user)
    # find channel by name or ID
    channel = None
    # try find by name
    for guild in client.guilds:
        for ch in guild.text_channels:
            if ch.name == DISCORD_CHANNEL:
                channel = ch
                break
        if channel:
            break
    # try interpret as ID
    if channel is None:
        try:
            ch_id = int(DISCORD_CHANNEL)
            channel = client.get_channel(ch_id)
        except Exception:
            pass
    if channel is None:
        # fallback to first text channel of first guild
        if client.guilds and client.guilds[0].text_channels:
            channel = client.guilds[0].text_channels[0]
            print(f"[warning] channel {DISCORD_CHANNEL} not found, using {channel.name}")
        else:
            print("[error] no channel found and no guilds accessible")
            return
    # start monitor
    client.loop.create_task(monitor_loop(channel))

# basic simple commands (optional)
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith("!price"):
        parts = message.content.split()
        if len(parts) >= 2:
            sym = parts[1].upper()
            try:
                df = yf.download(sym, period="1d", interval="1m", progress=False)
                price = df['Close'].iloc[-1]
                await message.channel.send(f"{sym} price: {fmt_price(price)}")
            except Exception as e:
                await message.channel.send(f"Erreur price {sym}: {e}")
    if message.content.startswith("!help"):
        await message.channel.send("Commands: `!price <SYMBOL>`")

# ========== RUN ==========
if __name__ == "__main__":
    try:
        client.run(DISCORD_TOKEN)
    except Exception as exc:
        print("Fatal error starting client:", exc)
