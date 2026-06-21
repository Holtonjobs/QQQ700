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
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")   # 可選，沒有則自動用 Google News RSS

# 高影響關鍵詞 (分級權重)
HIGH_IMPACT_KEYWORDS = {
    # 🔴 最高權重 3.0：極重大事件
    "聯儲局加息": 3.0,
    "fed rate hike": 3.0,
    "聯儲局減息": 3.0,
    "fed rate cut": 3.0,
    "縮表": 3.0,
    "量化緊縮": 3.0,
    "quantitative tightening": 3.0,
    "戰爭": 3.0,
    "war": 3.0,
    "金融危機": 3.0,
    "financial crisis": 3.0,
    "中國限制資金出境": 3.0,
    "china capital outflow": 3.0,
    "中美制裁": 3.0,
    "us china sanctions": 3.0,
    "台灣衝突": 3.0,
    "taiwan strait": 3.0,
    "銀行倒閉": 3.0,
    "bank failure": 3.0,

    # 🟠 次高權重 2.0：重大事件
    "spacex ipo": 2.0,
    "spacex上市": 2.0,
    "cpi 高於預期": 2.0,
    "cpi 低於預期": 2.0,
    "非農就業": 2.0,
    "nonfarm payroll": 2.0,
    "美聯儲主席": 2.0,
    "powell": 2.0,
    "中美貿易戰": 2.0,
    "us china trade war": 2.0,
    "關稅": 2.0,
    "tariff": 2.0,
    "晶片禁令": 2.0,
    "chip ban": 2.0,
    "騰訊大股東減持": 2.0,
    "prosus 減持": 2.0,
    "中概股審計": 2.0,
    "sec 中概": 2.0,
    "遊戲版號": 2.0,
    "反壟斷": 2.0,
    "antitrust": 2.0,
    "房地產危機": 2.0,
    "恒大": 2.0,
    "evergrande": 2.0,
    "降準": 2.0,
    "人民銀行降準": 2.0,
    "lpr 下調": 2.0,
    "ai 監管": 2.0,
    "ai regulation": 2.0,
    "科技巨頭財報": 2.0,
    "蘋果財報": 2.0,
    "apple earnings": 2.0,
    "微軟財報": 2.0,
    "microsoft earnings": 2.0,
    "英偉達財報": 2.0,
    "nvidia earnings": 2.0,
    "特斯拉財報": 2.0,
    "tesla earnings": 2.0,
    "半導體短缺": 2.0,
    "chip shortage": 2.0,

    # 🟡 一般權重 1.5：值得關注的事件
    "聯儲局會議": 1.5,
    "fomc": 1.5,
    "港股通": 1.5,
    "互聯互通": 1.5,
    "騰訊業績": 1.5,
    "tencent earnings": 1.5,
    "nasdaq 新高": 1.5,
    "納斯達克 新高": 1.5,
    "美聯儲鴿派": 1.5,
    "fed dovish": 1.5,
    "美聯儲鷹派": 1.5,
    "fed hawkish": 1.5,
    "債息倒掛": 1.5,
    "yield curve": 1.5,
    "衰退": 1.5,
    "recession": 1.5,
}

TICKERS = {
    "納斯達克100指數": "^NDX",
    "恒生指數": "^HSI"
}
TRADE_MAP = {"^NDX": "QQQ", "^HSI": "騰訊700"}
NEWS_QUERIES = {
    "^NDX": "NASDAQ 100 OR NDX OR QQQ OR 納斯達克100",
    "^HSI": "Hang Seng Index OR 恒生指數 OR 騰訊 OR Tencent"
}

# ----------------------------- 函式 -----------------------------
def get_pivot_signal(ticker):
    """計算前一日 Pivot 與技術信號"""
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if len(data) < 2:
            return None, None, None, None
        prev = data.iloc[-2]
        h, l, c = float(prev['High']), float(prev['Low']), float(prev['Close'])
        pivot = (h + l + c) / 3
        s1 = 2 * pivot - h
        r1 = 2 * pivot - l
        if c > pivot:
            signal = 1
        elif c < pivot:
            signal = -1
        else:
            signal = 0
        return pivot, (s1, r1), signal, (h, l, c)
    except Exception as e:
        print(f"獲取 {ticker} 失敗：{e}")
        return None, None, None, None

def fetch_news(query, from_date, to_date):
    """用 NewsAPI 或 Google News RSS 取得新聞標題"""
    if NEWSAPI_KEY:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "relevancy",
            "apiKey": NEWSAPI_KEY,
            "pageSize": 10
        }
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            articles = resp.json().get("articles", [])
            return [a["title"] for a in articles if a["title"]]
    # 後備：Google News RSS
    try:
        import feedparser
        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:10]]
    except:
        return []

def analyze_news(news_titles):
    """情感分析並加權，回傳 (平均情感分數, 詳細列表)"""
    nltk.download('vader_lexicon', quiet=True)
    sid = SentimentIntensityAnalyzer()
    news_details = []
    total_weight = 0
    weighted_score = 0
    for title in news_titles:
        weight = 1.0
        for kw, w in HIGH_IMPACT_KEYWORDS.items():
            if kw.lower() in title.lower():
                weight = w
                break
        score = sid.polarity_scores(title)['compound']
        weighted_score += score * weight
        total_weight += weight
        news_details.append((title, score, weight))
    if total_weight == 0:
        return 0, news_details
    return weighted_score / total_weight, news_details

def send_whatsapp_chunks(full_text, chunk_size=400):
    """將長文字分段，每段 chunk_size 字，用 WhatsApp 發送"""
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i+chunk_size]
        encoded = urllib.parse.quote(chunk)
        url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={encoded}&apikey={WHATSAPP_API_KEY}"
        try:
            requests.get(url, timeout=10)
        except Exception as e:
            print(f"WhatsApp 發送失敗: {e}")

def build_compact_report(ticker_name, ticker, trade_instrument, pivot_data, news_data):
    """產生精簡報告（適合 WhatsApp 傳送）"""
    today_str = datetime.date.today().isoformat()
    pivot, supports, pivot_signal, ohlc = pivot_data
    if pivot is None:
        return f"⚠️ {ticker_name} 數據缺失，無法預測"

    news_score, news_details = news_data
    # 新聞信號轉換
    if news_score > 0.15:
        news_signal = 1
    elif news_score < -0.15:
        news_signal = -1
    else:
        news_signal = 0

    final_score = 0.5 * pivot_signal + 0.5 * news_signal
    if final_score > 0.3:
        prediction = "上升📈"
    elif final_score < -0.3:
        prediction = "下跌📉"
    else:
        prediction = "震盪↔️"

    # 只列出高影響新聞 (權重 >= 2.0)
    highlights = [t for t in news_details if t[2] >= 2.0]
    high_news_str = ", ".join([f"「{t[0][:30]}」" for t in highlights]) if highlights else "無高影響新聞"

    report = (
        f"📅{today_str} {ticker_name}({trade_instrument})\n"
        f"預測：{prediction}\n"
        f"Pivot:{pivot:.0f} 前收:{ohlc[2]:.0f} 信號:{'多' if pivot_signal==1 else '空' if pivot_signal==-1 else '中'}\n"
        f"新聞情緒:{news_score:.2f} 信號:{news_signal}\n"
        f"高影響:{high_news_str}\n"
        f"理由:{'價格偏多+利多主導' if '上升' in prediction else '價格偏空+利空主導' if '下跌' in prediction else '多空拉鋸，方向不明'}"
    )
    return report

# ----------------------------- 主程式 -----------------------------
def main():
    # 命令列參數決定標的：HSI 或 NDX，無參數則全部執行
    target = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg == "HSI":
            target = "^HSI"
        elif arg == "NDX":
            target = "^NDX"

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    date_from = yesterday.isoformat()
    date_to = today.isoformat()

    for name, ticker in TICKERS.items():
        if target and ticker != target:
            continue

        # 1. 取得 Pivot 技術信號
        pivot_data = get_pivot_signal(ticker)
        if pivot_data[0] is None:
            send_whatsapp_chunks(f"⚠️ {name} 數據缺失，跳過今日預測")
            continue

        # 2. 取得新聞並情感分析
        query = NEWS_QUERIES.get(ticker, name)
        news_titles = fetch_news(query, date_from, date_to)
        if not news_titles:
            news_titles = ["無相關新聞"]
        news_score, news_details = analyze_news(news_titles)

        # 3. 生成報告並發送
        report = build_compact_report(name, ticker, TRADE_MAP[ticker], pivot_data, (news_score, news_details))
        send_whatsapp_chunks(report, chunk_size=400)

if __name__ == "__main__":
    main()
