import yfinance as yf
import requests
import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import urllib.parse
import sys

# ----------------------------- 設定 -----------------------------
WHATSAPP_PHONE = os.environ["WHATSAPP_PHONE"]
WHATSAPP_API_KEY = os.environ["WHATSAPP_API_KEY"]
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

TICKERS = {
    "QQQ": "QQQ",
    "騰訊700": "0700.HK"
}
TRADE_MAP = {"QQQ": "QQQ", "0700.HK": "騰訊700"}

NEWS_QUERIES = {
    "QQQ": "QQQ OR Nasdaq ETF OR US tech stocks",
    "0700.HK": "騰訊 OR Tencent OR 0700.HK"
}

HIGH_IMPACT_KEYWORDS = {
    "聯儲局加息": 3.0, "fed rate hike": 3.0,
    "聯儲局減息": 3.0, "fed rate cut": 3.0,
    "縮表": 3.0, "量化緊縮": 3.0, "quantitative tightening": 3.0,
    "戰爭": 3.0, "war": 3.0,
    "金融危機": 3.0, "financial crisis": 3.0,
    "中國限制資金出境": 3.0, "china capital outflow": 3.0,
    "中美制裁": 3.0, "us china sanctions": 3.0,
    "台灣衝突": 3.0, "taiwan strait": 3.0,
    "銀行倒閉": 3.0, "bank failure": 3.0,
    "spacex ipo": 2.0, "spacex上市": 2.0,
    "cpi 高於預期": 2.0, "cpi 低於預期": 2.0,
    "非農就業": 2.0, "nonfarm payroll": 2.0,
    "美聯儲主席": 2.0, "powell": 2.0,
    "中美貿易戰": 2.0, "us china trade war": 2.0,
    "關稅": 2.0, "tariff": 2.0,
    "晶片禁令": 2.0, "chip ban": 2.0,
    "騰訊大股東減持": 2.0, "prosus 減持": 2.0,
    "中概股審計": 2.0, "sec 中概": 2.0,
    "遊戲版號": 2.0, "反壟斷": 2.0, "antitrust": 2.0,
    "房地產危機": 2.0, "恒大": 2.0, "evergrande": 2.0,
    "降準": 2.0, "人民銀行降準": 2.0, "lpr 下調": 2.0,
    "ai 監管": 2.0, "ai regulation": 2.0,
    "科技巨頭財報": 2.0, "蘋果財報": 2.0, "apple earnings": 2.0,
    "微軟財報": 2.0, "microsoft earnings": 2.0,
    "英偉達財報": 2.0, "nvidia earnings": 2.0,
    "特斯拉財報": 2.0, "tesla earnings": 2.0,
    "半導體短缺": 2.0, "chip shortage": 2.0,
    "聯儲局會議": 1.5, "fomc": 1.5,
    "港股通": 1.5, "互聯互通": 1.5,
    "騰訊業績": 1.5, "tencent earnings": 1.5,
    "nasdaq 新高": 1.5, "納斯達克 新高": 1.5,
    "美聯儲鴿派": 1.5, "fed dovish": 1.5,
    "美聯儲鷹派": 1.5, "fed hawkish": 1.5,
    "債息倒掛": 1.5, "yield curve": 1.5,
    "衰退": 1.5, "recession": 1.5,
}

EVENTS = {
    '2026-06-19': ['四巫日', '美股期權結算'],
    '2026-06-22': ['FOMC會議紀錄公佈'],
    '2026-06-24': ['美國GDP終值'],
    '2026-07-01': ['香港回歸紀念日休市'],
}

def get_today_events():
    return EVENTS.get(datetime.date.today().isoformat(), [])

# ----------------------------- 工具函數 -----------------------------
def get_pivot_signal(ticker):
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if len(data) < 2:
            return None, None, None
        prev = data.iloc[-2]
        h, l, c = float(prev['High']), float(prev['Low']), float(prev['Close'])
        pivot = (h + l + c) / 3
        r1 = 2 * pivot - l
        r2 = pivot + (h - l)
        s1 = 2 * pivot - h
        s2 = pivot - (h - l)
        signal = 1 if c > pivot else (-1 if c < pivot else 0)
        return {
            'pivot': pivot, 'r1': r1, 'r2': r2,
            's1': s1, 's2': s2, 'close': c, 'signal': signal,
            'high': h, 'low': l
        }, signal, c
    except Exception as e:
        print(f"Pivot錯誤 {ticker}: {e}")
        return None, None, None

def fetch_news(query, from_date, to_date):
    if NEWSAPI_KEY:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query, "from": from_date, "to": to_date,
            "language": "en", "sortBy": "relevancy",
            "apiKey": NEWSAPI_KEY, "pageSize": 10
        }
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            articles = resp.json().get("articles", [])
            return [a["title"] for a in articles if a["title"]]
    try:
        import feedparser
        rss = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US"
        feed = feedparser.parse(rss)
        return [e.title for e in feed.entries[:10]]
    except:
        return []

def analyze_news(news_titles):
    nltk.download('vader_lexicon', quiet=True)
    sid = SentimentIntensityAnalyzer()
    details = []
    for title in news_titles:
        weight = 1.0
        for kw, w in HIGH_IMPACT_KEYWORDS.items():
            if kw.lower() in title.lower():
                weight = w
                break
        score = sid.polarity_scores(title)['compound']
        details.append((title, score, weight))
    if not details:
        return 0, [], []
    total_w = sum(d[2] for d in details)
    avg = sum(d[1]*d[2] for d in details) / total_w
    top3 = sorted(details, key=lambda x: abs(x[1]*x[2]), reverse=True)[:3]
    return avg, details, top3

def get_fundamental_signal(ticker):
    """
    取得簡易基本面信號:
    signal: -1 (高估), 0 (合理), 1 (低估)
    text: 簡短描述
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get('forwardPE') or info.get('trailingPE')
        pb = info.get('priceToBook')
        div_yield = info.get('dividendYield')

        # 根據標的設定不同的閾值
        if ticker == "QQQ":
            # QQQ 是科技成長型，PE 通常較高
            pe_low, pe_high = 25, 35
            pb_high = 8
        else:  # 騰訊
            pe_low, pe_high = 15, 25
            pb_high = 5

        signals = []
        details = []

        if pe:
            details.append(f"PE:{pe:.1f}")
            if pe < pe_low:
                signals.append(1)
            elif pe > pe_high:
                signals.append(-1)
            else:
                signals.append(0)

        if pb:
            details.append(f"PB:{pb:.1f}")
            if pb > pb_high:
                signals.append(-1)
            else:
                signals.append(0)

        if div_yield and div_yield > 0.03:
            details.append(f"息率:{div_yield*100:.1f}%")
            signals.append(1)

        if not signals:
            return 0, "無足夠數據"

        avg_signal = sum(signals) / len(signals)
        if avg_signal > 0.3:
            signal, ver = 1, "低估"
        elif avg_signal < -0.3:
            signal, ver = -1, "高估"
        else:
            signal, ver = 0, "合理"

        text = f"{ver} ({', '.join(details)})"
        return signal, text
    except Exception as e:
        print(f"基本面取得失敗 {ticker}: {e}")
        return 0, "數據缺失"

def send_whatsapp(text):
    encoded = urllib.parse.quote(text)
    url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={encoded}&apikey={WHATSAPP_API_KEY}"
    try:
        requests.get(url, timeout=10)
    except Exception as e:
        print(f"發送失敗: {e}")

def send_report_safe(report, max_chars=1400):
    if len(report) <= max_chars:
        send_whatsapp(report)
    else:
        for i in range(0, len(report), max_chars):
            send_whatsapp(report[i:i+max_chars])

# ----------------------------- 報告產生 -----------------------------
def build_report(name, ticker, trade_inst, pivot_data, news_avg, top3_news, events, fund_signal, fund_text):
    today = datetime.date.today()
    if not pivot_data:
        return None

    p = pivot_data
    price = p['close']
    pivot = p['pivot']
    r1, r2 = p['r1'], p['r2']
    s1, s2 = p['s1'], p['s2']

    news_signal = 1 if news_avg > 0.15 else (-1 if news_avg < -0.15 else 0)

    # 綜合分數：技術 30% + 新聞 40% + 基本面 30%
    final_score = 0.3 * p['signal'] + 0.4 * news_signal + 0.3 * fund_signal

    event_str = ""
    if events:
        event_str = "⚠️ 今日事件: " + ", ".join(events) + "\n"

    news_lines = []
    for title, score, weight in top3_news:
        emoji = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
        news_lines.append(f"{emoji} [{weight}x] {title[:80]}")

    # 單一最可能預測與操作
    if final_score > 0.4:
        prediction = "📈 上升 (做多)"
        plan = f"操作：突破 R1 {r1:.2f} 後做多，目標 R2 {r2:.2f}，止損設 Pivot {pivot:.2f}"
    elif final_score < -0.4:
        prediction = "📉 下跌 (做空)"
        plan = f"操作：跌破 S1 {s1:.2f} 後做空，目標 S2 {s2:.2f}，止損設 Pivot {pivot:.2f}"
    else:
        prediction = "↔️ 震盪 (區間交易)"
        plan = f"操作：於 S1 {s1:.2f} 附近做多，R1 {r1:.2f} 附近做空，止損設區間外 S2 {s2:.2f} / R2 {r2:.2f}"

    report = (
        f"📅 {today.isoformat()} | {name} ({trade_inst})\n"
        f"{event_str}"
        f"💰 前收: {price:.2f}  樞軸: {pivot:.2f}\n"
        f"📊 基本面: {fund_text}\n"
        f"🗞️ 新聞情緒: {news_avg:.2f}\n"
        f"--- 關鍵新聞 ---\n" + "\n".join(news_lines) + "\n"
        f"🎯 預測: {prediction}\n"
        f"{plan}"
    )
    return report

# ----------------------------- 主流程 -----------------------------
def main():
    target = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg == "QQQ":
            target = "QQQ"
        elif arg == "700":
            target = "0700.HK"

    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=7)
    events = get_today_events()

    for name, ticker in TICKERS.items():
        if target and ticker != target:
            continue

        pivot_data, _, _ = get_pivot_signal(ticker)
        if not pivot_data:
            send_whatsapp(f"⚠️ {name} 數據缺失")
            continue

        query = NEWS_QUERIES.get(ticker, name)
        titles = fetch_news(query, week_ago.isoformat(), today.isoformat())
        if not titles:
            titles = ["無相關新聞"]
        news_avg, _, top3 = analyze_news(titles)

        fund_signal, fund_text = get_fundamental_signal(ticker)

        report = build_report(name, ticker, TRADE_MAP.get(ticker, name),
                             pivot_data, news_avg, top3, events,
                             fund_signal, fund_text)
        if report:
            send_report_safe(report)

if __name__ == "__main__":
    main()
