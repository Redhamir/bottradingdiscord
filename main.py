import os
import discord
from discord.ext import commands
import yfinance as yf
import pandas as pd
from datetime import datetime

# VOTRE WATCHLIST PERSONNALISÉE
WATCHLIST = {
    'CRYPTO': ['BTC-USD', 'ETH-USD', 'XLM-USD', 'ADA-USD', 'XTZ-USD', 'LTC-USD', 'DOT-USD', 'LINK-USD'],
    'ACTIONS': ['TSLA', 'AAPL', 'ORCL', 'NVDA', 'V', 'AMD', 'GOOGL'],
    'INDICES': ['SPY', 'QQQ', 'GC=F']
}

# Configuration du bot Discord
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

class TradingBot:
    def __init__(self):
        self.price_cache = {}
    
    def get_price(self, symbol):
        """Récupère le prix d'un symbole"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if len(data) > 0:
                return data['Close'].iloc[-1]
        except Exception as e:
            print(f"Erreur pour {symbol}: {e}")
        return None

# Initialisation
trading_bot = TradingBot()

@bot.event
async def on_ready():
    print(f'✅ {bot.user.name} est en ligne sur Render!')
    activity = discord.Activity(type=discord.ActivityType.watching, name="!help 📈")
    await bot.change_presence(activity=activity)

@bot.command(name='help')
async def help_command(ctx):
    """Affiche toutes les commandes"""
    embed = discord.Embed(
        title="🤖 BOT TRADING DISCORD",
        description="**Hébergé sur Render.com - 24h/24 Gratuit**\n\nCommandes disponibles :",
        color=0x7289da
    )
    
    commands_list = [
        ("!prix [symbole]", "Affiche le prix d'un actif\nEx: `!prix BTC-USD`"),
        ("!watchlist", "Votre watchlist complète"),
        ("!crypto", "Cryptomonnaies seulement"),
        ("!actions", "Actions seulement"),
        ("!analyse [symbole]", "Analyse technique RSI"),
        ("!info", "Informations du bot")
    ]
    
    for cmd, desc in commands_list:
        embed.add_field(name=cmd, value=desc, inline=False)
    
    embed.set_footer(text="🚀 Service 100% gratuit - Disponible 24h/24")
    await ctx.send(embed=embed)

@bot.command(name='info')
async def info(ctx):
    """Informations sur le bot"""
    embed = discord.Embed(
        title="ℹ️ INFORMATIONS DU BOT",
        color=0x00ff00
    )
    
    embed.add_field(name="🏠 Hébergement", value="Render.com", inline=True)
    embed.add_field(name="⏰ Disponibilité", value="24h/24 - 7j/7", inline=True)
    embed.add_field(name="💰 Coût", value="100% Gratuit", inline=True)
    embed.add_field(name="📊 Actifs suivis", value="16 actifs", inline=True)
    embed.add_field(name="🆓 Plan", value="Free Tier", inline=True)
    embed.add_field(name="🚀 Statut", value="✅ En ligne", inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name='prix')
async def prix(ctx, *, symbole: str = "BTC-USD"):
    """Affiche le prix d'un actif - !prix BTC-USD"""
    symbol = symbole.upper().strip()
    price = trading_bot.get_price(symbol)
    
    if price and price > 0:
        embed = discord.Embed(
            title=f"💰 {symbol}",
            description=f"**${price:,.2f}**",
            color=0x00ff00,
            timestamp=datetime.now()
        )
        embed.set_footer(text="Données en temps réel via yfinance")
    else:
        embed = discord.Embed(
            title="❌ Symbole non trouvé",
            description=f"**{symbol}** n'a pas pu être trouvé\nVérifiez le symbole et réessayez.",
            color=0xff0000
        )
    
    await ctx.send(embed=embed)

@bot.command(name='watchlist')
async def watchlist(ctx):
    """Affiche toute votre watchlist"""
    embed = discord.Embed(
        title="📊 VOTRE WATCHLIST COMPLÈTE",
        description="**Vos actifs préférés en temps réel**",
        color=0x0099ff,
        timestamp=datetime.now()
    )
    
    # Cryptomonnaies
    crypto_text = ""
    for symbol in WATCHLIST['CRYPTO']:
        price = trading_bot.get_price(symbol)
        if price:
            crypto_text += f"• **{symbol}** : ${price:,.2f}\n"
    
    if crypto_text:
        embed.add_field(
            name="🪙 CRYPTOMONNAIES",
            value=crypto_text,
            inline=False
        )
    
    # Actions
    actions_text = ""
    for symbol in WATCHLIST['ACTIONS']:
        price = trading_bot.get_price(symbol)
        if price:
            actions_text += f"• **{symbol}** : ${price:,.2f}\n"
    
    if actions_text:
        embed.add_field(
            name="📈 ACTIONS",
            value=actions_text,
            inline=False
        )
    
    # Indices
    indices_text = ""
    for symbol in WATCHLIST['INDICES']:
        price = trading_bot.get_price(symbol)
        if price:
            indices_text += f"• **{symbol}** : ${price:,.2f}\n"
    
    if indices_text:
        embed.add_field(
            name="📊 INDICES & MÉTAUX",
            value=indices_text,
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name='crypto')
async def crypto(ctx):
    """Prix des cryptomonnaies seulement"""
    embed = discord.Embed(
        title="🪙 CRYPTOMONNAIES",
        color=0xf7931a,
        timestamp=datetime.now()
    )
    
    for symbol in WATCHLIST['CRYPTO']:
        price = trading_bot.get_price(symbol)
        if price:
            embed.add_field(name=symbol, value=f"${price:,.2f}", inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name='actions')
async def actions(ctx):
    """Prix des actions seulement"""
    embed = discord.Embed(
        title="📈 ACTIONS",
        color=0x00ff00,
        timestamp=datetime.now()
    )
    
    for symbol in WATCHLIST['ACTIONS']:
        price = trading_bot.get_price(symbol)
        if price:
            embed.add_field(name=symbol, value=f"${price:,.2f}", inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name='analyse')
async def analyse(ctx, *, symbole: str = "BTC-USD"):
    """Analyse technique avec RSI - !analyse TSLA"""
    symbol = symbole.upper().strip()
    
    try:
        # Récupération des données
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='5d', interval='1h')
        
        if len(data) < 24:
            await ctx.send(f"❌ Données insuffisantes pour analyser **{symbol}**")
            return
        
        current_price = data['Close'].iloc[-1]
        price_24h_ago = data['Close'].iloc[-24] if len(data) >= 24 else data['Close'].iloc[0]
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        
        # Calcul du RSI
        import ta
        rsi = ta.momentum.RSIIndicator(data['Close'], window=14).rsi().iloc[-1]
        
        # Création de l'embed
        embed = discord.Embed(
            title=f"🔍 ANALYSE TECHNIQUE - {symbol}",
            color=0xff9900,
            timestamp=datetime.now()
        )
        
        embed.add_field(name="💰 PRIX ACTUEL", value=f"${current_price:,.2f}", inline=True)
        embed.add_field(name="📊 VARIATION 24H", value=f"{change_24h:+.2f}%", inline=True)
        embed.add_field(name="📈 RSI (14)", value=f"{rsi:.1f}", inline=True)
        
        # Recommandation basée sur RSI
        if rsi < 30:
            recommendation = "🟢 **OVERSOLD** - Potentiel d'achat"
            embed.color = 0x00ff00
        elif rsi > 70:
            recommendation = "🔴 **OVERBOUGHT** - Attention surachat"
            embed.color = 0xff0000
        else:
            recommendation = "⚪ **ZONE NEUTRE** - Attendre un signal"
        
        embed.add_field(
            name="💡 RECOMMANDATION RSI", 
            value=recommendation,
            inline=False
        )
        
        # Interprétation RSI
        if rsi < 30:
            interpretation = "Le RSI indique une condition de suvente. Potentiel rebussement."
        elif rsi > 70:
            interpretation = "Le RSI indique une condition de surachat. Correction possible."
        else:
            interpretation = "Le RSI est en zone neutre. Aucun signal fort."
        
        embed.add_field(
            name="📖 INTERPRÉTATION",
            value=interpretation,
            inline=False
        )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Erreur lors de l'analyse de **{symbol}** : {str(e)}")

# Lancement du bot
if __name__ == "__main__":
    bot.run(os.environ['DISCORD_TOKEN'])