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

# 高影響關鍵詞（分級權重）
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

TICKERS = {"納斯達克100指數": "^NDX", "恒生指數": "^HSI"}
TRADE_MAP = {"^NDX": "QQQ", "^HSI": "騰訊700"}
NEWS_QUERIES = {
    "^NDX": "NASDAQ 100 OR NDX OR QQQ OR 納斯達克100",
    "^HSI": "Hang Seng Index OR 恒生指數 OR 騰訊 OR Tencent"
}

# ----------------------------- 工具函數 -----------------------------
def get_multi_pivot(ticker):
    """
    回傳 dict:
      daily: {pivot, s1, s2, r1, r2, signal, close}
      hourly: {pivot, s1, s2, r1, r2, signal, close}
    """
    result = {}
    try:
        # 日線
        d = yf.download(ticker, period="5d", interval="1d", progress=False)
        if len(d) >= 2:
            prev = d.iloc[-2]
            h, l, c = float(prev['High']), float(prev['Low']), float(prev['Close'])
            p = (h + l + c) / 3
            result['daily'] = {
                'pivot': p,
                'r1': 2*p - l, 'r2': p + (h - l),
                's1': 2*p - h, 's2': p - (h - l),
                'signal': 1 if c > p else (-1 if c < p else 0),
                'close': c, 'high': h, 'low': l
            }
        # 1小時線
        h_data = yf.download(ticker, period="5d", interval="1h", progress=False)
        if len(h_data) >= 2:
            prev_h = h_data.iloc[-2]
            hh, hl, hc = float(prev_h['High']), float(prev_h['Low']), float(prev_h['Close'])
            hp = (hh + hl + hc) / 3
            result['hourly'] = {
                'pivot': hp,
                'r1': 2*hp - hl, 'r2': hp + (hh - hl),
                's1': 2*hp - hh, 's2': hp - (hh - hl),
                'signal': 1 if hc > hp else (-1 if hc < hp else 0),
                'close': hc, 'high': hh, 'low': hl
            }
    except Exception as e:
        print(f"Pivot 錯誤: {e}")
    return result

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
    # 後備 RSS
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
    total_w = sum(d[2] for d in details)
    avg = sum(d[1]*d[2] for d in details) / total_w if total_w else 0
    top3 = sorted(details, key=lambda x: abs(x[1]*x[2]), reverse=True)[:3]
    return avg, details, top3

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

# ----------------------------- 報告產生（含目標價位） -----------------------------
def build_enhanced_report(name, ticker, trade_inst, pivots, news_avg, top3_news):
    today = datetime.date.today().isoformat()
    daily = pivots.get('daily')
    hourly = pivots.get('hourly')
    if not daily:
        return None

    price = daily['close']
    # 多週期狀態
    pivot_status = []
    if daily:
        pos = []
        if price > daily['r1']: pos.append("破R1")
        elif price > daily['pivot']: pos.append("站上Pivot")
        elif price > daily['s1']: pos.append("守S1")
        elif price > daily['s2']: pos.append("測S2")
        else: pos.append("破S2")
        pivot_status.append(f"日:{'/'.join(pos)}")
    if hourly:
        hpos = []
        if price > hourly['r1']: hpos.append("破R1")
        elif price > hourly['pivot']: hpos.append("站上Pivot")
        elif price > hourly['s1']: hpos.append("守S1")
        elif price > hourly['s2']: hpos.append("測S2")
        else: hpos.append("破S2")
        pivot_status.append(f"時:{'/'.join(hpos)}")

    # 綜合信號
    d_signal = daily['signal']
    h_signal = hourly['signal'] if hourly else 0
    news_signal = 1 if news_avg > 0.15 else (-1 if news_avg < -0.15 else 0)
    final_score = 0.3*d_signal + 0.2*h_signal + 0.5*news_signal

    # 目標價位 (基於日線Pivot)
    r1, r2 = daily['r1'], daily['r2']
    s1, s2 = daily['s1'], daily['s2']
    target_text = ""
    if final_score > 0.3:
        prediction = "📈 上升"
        target_text = f"從 {price:.0f} 升至\n阻力: {r1:.0f}(R1) / {r2:.0f}(R2)"
        reason = "技術偏多+新聞正面"
    elif final_score < -0.3:
        prediction = "📉 下跌"
        target_text = f"從 {price:.0f} 跌至\n支撐: {s1:.0f}(S1) / {s2:.0f}(S2)"
        reason = "技術偏空+新聞負面"
    else:
        prediction = "↔️ 震盪"
        target_text = f"震盪區間\n支撐 {s1:.0f}(S1)~{s2:.0f}(S2)  阻力 {r1:.0f}(R1)~{r2:.0f}(R2)"
        reason = "多空拉鋸，方向不明"

    # 新聞亮點
    news_lines = []
    for title, score, weight in top3_news:
        emoji = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
        news_lines.append(f"{emoji} [{weight}x] {title[:80]}")

    report = (
        f"📅{today} {name}({trade_inst})\n"
        f"預測：{prediction}\n"
        f"{target_text}\n"
        f"Pivot狀態：{'，'.join(pivot_status)}\n"
        f"新聞情緒：{news_avg:.2f} (信號:{news_signal})\n"
        f"--- 最關鍵新聞 ---\n" +
        "\n".join(news_lines) + "\n"
        f"總結：{reason}，綜合分數 {final_score:.2f}"
    )
    return report

# ----------------------------- 主流程 -----------------------------
def main():
    target = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg == "HSI": target = "^HSI"
        elif arg == "NDX": target = "^NDX"

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    for name, ticker in TICKERS.items():
        if target and ticker != target:
            continue
        pivots = get_multi_pivot(ticker)
        if not pivots.get('daily'):
            send_whatsapp(f"⚠️ {name} 資料不足")
            continue
        query = NEWS_QUERIES.get(ticker, name)
        titles = fetch_news(query, yesterday.isoformat(), today.isoformat())
        if not titles:
            titles = ["無相關新聞"]
        news_avg, _, top3 = analyze_news(titles)
        report = build_enhanced_report(name, ticker, TRADE_MAP[ticker],
                                       pivots, news_avg, top3)
        if report:
            send_report_safe(report)

if __name__ == "__main__":
    main()
