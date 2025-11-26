import os
import discord
from discord.ext import commands, tasks
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask
from threading import Thread
import asyncio
import json
import logging
import time
import ta

# ------------------ CONFIG ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('trading-bot')

app = Flask('')

@app.route('/')
def home():
    return "🤖 Bot Trading Discord - EN LIGNE"

@app.route('/health')
def health():
    return {"status": "online", "service": "trading-bot"}

def run_flask_app():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

# ------------------ WATCHLIST ------------------
WATCHLIST = {
    'CRYPTO': ['BTC-USD', 'ETH-USD', 'XLM-USD', 'ADA-USD', 'XTZ-USD', 'LTC-USD', 'DOT-USD', 'LINK-USD'],
    'ACTIONS': ['TSLA', 'AAPL', 'ORCL', 'NVDA', 'V', 'AMD', 'GOOGL'],
    'INDICES': ['SPY', 'QQQ', 'GC=F']
}

ALERTS_FILE = 'alerts.json'
alerts_lock = asyncio.Lock()

# ------------------ DISCORD BOT ------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# ------------------ TRADING BOT ------------------
class TradingBot:
    def __init__(self):
        self.price_cache = {}

    def get_price(self, symbol: str):
        """Récupère le prix d'un symbole via yfinance avec cache TTL 30s."""
        symbol = symbol.upper().strip()
        cached = self.price_cache.get(symbol)
        if cached and time.time() - cached[1] < 30:
            return cached[0]
        try:
            data = yf.Ticker(symbol).history(period='2d', interval='1m', actions=False)
            if data is None or data.empty:
                return None
            close = data['Close'].dropna()
            if close.empty:
                return None
            price = float(close.iloc[-1])
            self.price_cache[symbol] = (price, time.time())
            return price
        except Exception:
            logger.exception(f"Erreur get_price {symbol}")
            return None

trading_bot = TradingBot()

# ------------------ ALERTS ------------------
def load_alerts():
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            logger.exception('Impossible de charger alerts.json')
    return []

async def save_alerts(alerts):
    async with alerts_lock:
        try:
            with open(ALERTS_FILE, 'w') as f:
                json.dump(alerts, f, indent=2)
        except Exception:
            logger.exception('Impossible de sauver alerts.json')

alerts = load_alerts()
next_alert_id = max([a['id'] for a in alerts], default=0) + 1

# ------------------ ALERT CHECKERS ------------------
async def check_price_alert(alert):
    price = trading_bot.get_price(alert['symbol'])
    if price is None:
        return False, None
    op = alert.get('operator')
    val = float(alert.get('value'))
    if op == 'above' and price > val:
        return True, price
    if op == 'below' and price < val:
        return True, price
    return False, price

async def check_change_alert(alert):
    symbol = alert['symbol']
    period_h = int(alert.get('period_h', 24))
    period_h = min(period_h, 720)  # limitation Yahoo
    try:
        data = yf.Ticker(symbol).history(period=f'{period_h + 1}h', interval='1h')
        close = data['Close'].dropna()
        if len(close) < 2:
            return False, None
        current, past = float(close.iloc[-1]), float(close.iloc[0])
        change = ((current - past) / past) * 100
        op = alert.get('operator')
        tgt = float(alert.get('value'))
        if op == 'above' and change > tgt:
            return True, change
        if op == 'below' and change < -abs(tgt):
            return True, change
        return False, change
    except Exception:
        logger.exception('Erreur check_change_alert')
        return False, None

async def check_volume_alert(alert):
    symbol = alert['symbol']
    try:
        data = yf.Ticker(symbol).history(period='60d', interval='1d')
        vol_series = data['Volume'].dropna()
        if vol_series.empty:
            return False, None
        vol_today = float(vol_series.iloc[-1])
        avg_vol = vol_series[-20:].mean() if len(vol_series) >= 20 else vol_series.mean()
        multiplier = float(alert.get('value', 2.0))
        if avg_vol == 0:
            return False, None
        if vol_today > avg_vol * multiplier:
            return True, {'vol_today': vol_today, 'avg_vol': avg_vol}
        return False, {'vol_today': vol_today, 'avg_vol': avg_vol}
    except Exception:
        logger.exception('Erreur check_volume_alert')
        return False, None

async def check_event_alert(alert):
    symbol = alert['symbol']
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        if cal is None or cal.empty:
            return False, "Aucun événement récent"
        cal_str = cal.astype(str).to_string()
        return True, cal_str
    except Exception:
        logger.exception('Erreur check_event_alert')
        return False, "Erreur récupération events"

# ------------------ ALERT LOOP ------------------
@tasks.loop(seconds=60.0)
async def alert_checker_loop():
    global alerts
    if not alerts:
        return
    logger.info('Vérification des alertes: %d alertes', len(alerts))
    to_remove = []
    for alert in alerts:
        try:
            alert_type = alert.get('type')
            chat_id = alert.get('chat_id')
            triggered, detail = False, None
            if alert_type == 'price':
                triggered, detail = await check_price_alert(alert)
            elif alert_type == 'change':
                triggered, detail = await check_change_alert(alert)
            elif alert_type == 'volume':
                triggered, detail = await check_volume_alert(alert)
            elif alert_type == 'event':
                triggered, detail = await check_event_alert(alert)
            if triggered:
                channel = bot.get_channel(chat_id)
                if channel:
                    # message tronqué à 1900 caractères
                    msg = f"🔔 ALERTE {alert_type.upper()} {alert['symbol']} : {detail}"
                    if len(msg) > 1900:
                        msg = msg[:1897] + "..."
                    await channel.send(msg)
                if alert.get('once', True):
                    to_remove.append(alert['id'])
        except Exception:
            logger.exception('Erreur boucle alert_checker_loop')
    if to_remove:
        alerts = [a for a in alerts if a['id'] not in to_remove]
        await save_alerts(alerts)

@alert_checker_loop.before_loop
async def before_alerts():
    await bot.wait_until_ready()
    logger.info('Alert checker: bot ready, loop démarré')

# ------------------ COMMANDS ------------------
@bot.command(name='help')
async def help_command(ctx):
    embed = discord.Embed(
        title="🤖 BOT TRADING DISCORD",
        description="Commandes disponibles :",
        color=0x7289da
    )
    commands_list = [
        ("!prix [symbole]", "Affiche le prix d'un actif"),
        ("!watchlist", "Votre watchlist complète"),
        ("!alert add price SYMBOL above|below VALUE", "Ajouter alerte prix"),
        ("!alert add change SYMBOL above|below PERCENT [hours]", "Alerte % changement"),
        ("!alert add volume SYMBOL multiplier", "Alerte volume"),
        ("!alert list", "Lister vos alertes"),
        ("!alert remove ID", "Supprimer alerte")
    ]
    for cmd, desc in commands_list:
        embed.add_field(name=cmd, value=desc, inline=False)
    await ctx.send(embed=embed)

@bot.command(name='prix')
async def prix(ctx, *, symbole: str = "BTC-USD"):
    price = trading_bot.get_price(symbole)
    if price:
        await ctx.send(f"💰 {symbole.upper()}: ${price:,.2f}")
    else:
        await ctx.send(f"❌ Symbole {symbole.upper()} non trouvé")

# ------------------ ALERT COMMANDS ------------------
@bot.group(name='alert', invoke_without_command=True)
async def alert_group(ctx):
    await ctx.send('Usage: !alert add|list|remove')

@alert_group.command(name='add')
async def alert_add(ctx, subcommand: str, *args):
    global next_alert_id, alerts
    try:
        subcommand = subcommand.lower()
        if subcommand == 'price':
            sym, op, val = args[0].upper(), args[1].lower(), float(args[2])
            if op not in ['above', 'below']:
                await ctx.send("❌ Operator invalide, utilisez above ou below")
                return
            a = {'id': next_alert_id, 'type': 'price', 'symbol': sym, 'operator': op, 'value': val, 'chat_id': ctx.channel.id, 'once': True}
        elif subcommand == 'change':
            sym, op, val = args[0].upper(), args[1].lower(), float(args[2])
            if op not in ['above', 'below']:
                await ctx.send("❌ Operator invalide, utilisez above ou below")
                return
            hours = int(args[3]) if len(args) > 3 else 24
            a = {'id': next_alert_id, 'type': 'change', 'symbol': sym, 'operator': op, 'value': val, 'period_h': hours, 'chat_id': ctx.channel.id, 'once': True}
        elif subcommand == 'volume':
            sym, mult = args[0].upper(), float(args[1])
            a = {'id': next_alert_id, 'type': 'volume', 'symbol': sym, 'value': mult, 'chat_id': ctx.channel.id, 'once': True}
        else:
            await ctx.send('Sous-commande inconnue. Utilisez price|change|volume')
            return
        alerts.append(a)
        next_alert_id += 1
        await save_alerts(alerts)
        await ctx.send(f"✅ Alerte créée (ID: {a['id']}) pour {a['symbol']}")
    except Exception as e:
        logger.exception('Erreur creation alerte')
        await ctx.send(f"❌ Erreur création alerte: {e}")

@alert_group.command(name='list')
async def alert_list(ctx):
    if not alerts:
        await ctx.send('Aucune alerte enregistrée.')
        return
    lines = [f"ID {a['id']} - {a['type']} - {a['symbol']} - {a.get('operator','')}{a.get('value','')}" for a in alerts]
    msg = "\n".join(lines)
    if len(msg) > 1900:
        msg = msg[:1897] + "..."
    await ctx.send(f"📋 Alertes:\n{msg}")

@alert_group.command(name='remove')
async def alert_remove(ctx, alert_id: int):
    global alerts
    before = len(alerts)
    alerts = [a for a in alerts if a['id'] != alert_id]
    await save_alerts(alerts)
    if len(alerts) < before:
        await ctx.send(f"✅ Alerte {alert_id} supprimée")
    else:
        await ctx.send(f"❌ Alerte {alert_id} non trouvée")

# ------------------ ANALYSE RSI ------------------
@bot.command(name='analyse')
async def analyse(ctx, *, symbole: str = "BTC-USD"):
    symbol = symbole.upper().strip()
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='10d', interval='1h')
        close_series = data['Close'].dropna()
        if len(close_series) < 14:
            await ctx.send(f"❌ Pas assez de données pour RSI sur {symbol}")
            return
        rsi = ta.momentum.RSIIndicator(close_series, window=14).rsi().iloc[-1]
        current_price = float(close_series.iloc[-1])
        price_24h_ago = float(close_series.iloc[-24]) if len(close_series) >= 24 else float(close_series.iloc[0])
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100

        embed = discord.Embed(
            title=f"🔍 ANALYSE TECHNIQUE - {symbol}",
            color=0xff9900,
            timestamp=datetime.now()
        )
        embed.add_field(name="💰 PRIX ACTUEL", value=f"${current_price:,.2f}", inline=True)
        embed.add_field(name="📊 VARIATION 24H", value=f"{change_24h:+.2f}%", inline=True)
        embed.add_field(name="📈 RSI (14)", value=f"{rsi:.1f}", inline=True)

        if rsi < 30:
            rec = "🟢 OVERSOLD - Potentiel d'achat"
            embed.color = 0x00ff00
        elif rsi > 70:
            rec = "🔴 OVERBOUGHT - Attention surachat"
            embed.color = 0xff0000
        else:
            rec = "⚪ ZONE NEUTRE - Attendre signal"
        embed.add_field(name="💡 RECOMMANDATION RSI", value=rec, inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        logger.exception('Erreur analyse RSI')
        await ctx.send(f"❌ Erreur lors de l'analyse de {symbol}: {e}")

# ------------------ STARTUP ------------------
@bot.event
async def on_ready():
    logger.info(f'✅ {bot.user.name} est en ligne!')
    activity = discord.Activity(type=discord.ActivityType.watching, name="!help 📈")
    await bot.change_presence(activity=activity)
    if not alert_checker_loop.is_running():
        alert_checker_loop.start()

if __name__ == "__main__":
    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    token = os.environ.get('DISCORD_TOKEN')
    if token is None:
        logger.error("❌ ERREUR: DISCORD_TOKEN non défini")
    else:
        bot.run(token)
