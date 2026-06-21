import yfinance as yf
import requests
import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import urllib.parse

# ----------------------------- 設定 -----------------------------
WHATSAPP_PHONE = os.environ["WHATSAPP_PHONE"]
WHATSAPP_API_KEY = os.environ["WHATSAPP_API_KEY"]
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")   # 可選，沒有則用 RSS

# 高影響關鍵詞及權重
HIGH_IMPACT_KEYWORDS = {
    "spacex ipo": 2.0,
    "spacex上市": 2.0,
    "中國限制資金出境": 2.0,
    "china capital outflow": 2.0,
    "聯儲局加息": 2.0,
    "fed rate hike": 2.0,
    "騰訊大股東減持": 2.0,
}

TICKERS = {"納斯達克指數": "^IXIC", "恒生指數": "^HSI"}
TRADE_MAP = {"^IXIC": "QQQ", "^HSI": "騰訊700"}

# ----------------------------- 函式 -----------------------------
def get_pivot_signal(ticker):
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
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:10]]
    except:
        return []

def analyze_news(news_list):
    nltk.download('vader_lexicon', quiet=True)
    sid = SentimentIntensityAnalyzer()
    news_details = []
    total_weight = 0
    weighted_score = 0
    for title in news_list:
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
    avg_score = weighted_score / total_weight
    return avg_score, news_details

def send_whatsapp_message(text):
    phone = WHATSAPP_PHONE
    apikey = WHATSAPP_API_KEY
    encoded_text = urllib.parse.quote(text)
    url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={encoded_text}&apikey={apikey}"
    requests.get(url)

def build_report(ticker_name, ticker, trade_instrument, pivot_data, news_data):
    today_str = datetime.date.today().isoformat()
    pivot, supports, pivot_signal, ohlc = pivot_data
    if pivot is None:
        return f"⚠️ 無法計算 {ticker_name} Pivot，跳過。"
    
    news_score, news_details = news_data
    if news_score > 0.15:
        news_signal = 1
    elif news_score < -0.15:
        news_signal = -1
    else:
        news_signal = 0
    
    final_score = 0.5 * pivot_signal + 0.5 * news_signal
    if final_score > 0.3:
        prediction = "上升"
    elif final_score < -0.3:
        prediction = "下跌"
    else:
        prediction = "震盪"
    
    report = f"📅 預測日期：{today_str}\n"
    report += f"🎯 標的：{ticker_name} ({trade_instrument})\n"
    report += f"📈 預測結果：{prediction}\n\n"
    
    report += f"🔹 Pivot Point 分析\n"
    report += f"- 前日高/低/收：{ohlc[0]:.2f} / {ohlc[1]:.2f} / {ohlc[2]:.2f}\n"
    report += f"- Pivot：{pivot:.2f}， S1：{supports[0]:.2f}， R1：{supports[1]:.2f}\n"
    report += f"- 技術信號：{'偏多 (+1)' if pivot_signal==1 else '偏空 (-1)' if pivot_signal==-1 else '中性 (0)'}\n\n"
    
    report += f"🔸 新聞分析\n"
    for title, score, weight in news_details:
        flag = "🔥" if weight > 1 else ""
        report += f"{flag} \"{title[:60]}...\" (情感 {score:.2f}, 權重 {weight})\n"
    report += f"綜合新聞情感得分：{news_score:.2f}， 新聞信號：{news_signal}\n\n"
    
    reason = f"🧠 辯證：結合技術面與新聞面，"
    if prediction == "上升":
        reason += "價格結構偏多且利多消息主導，因此預測上升。"
    elif prediction == "下跌":
        reason += "價格弱勢配合利空新聞，因此預測下跌。"
    else:
        reason += "多空因素勢均力敵，方向不明，預測震盪。"
    report += reason
    return report

def main():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    date_from = yesterday.isoformat()
    date_to = today.isoformat()
    
    final_messages = []
    for name, ticker in TICKERS.items():
        pivot_data = get_pivot_signal(ticker)
        if pivot_data[0] is None:
            continue
        
        if "納斯達克" in name:
            query = "NASDAQ OR US tech stocks OR 美股"
        else:
            query = "Hang Seng Index OR 恒生指數 OR 港股"
        
        news_titles = fetch_news(query, date_from, date_to)
        if "恒生" in name:
            extra = fetch_news("騰訊 OR Tencent", date_from, date_to)
            news_titles.extend(extra)
        
        if not news_titles:
            news_titles = ["無相關新聞"]
        news_score, news_details = analyze_news(news_titles)
        
        report = build_report(name, ticker, TRADE_MAP[ticker], pivot_data, (news_score, news_details))
        final_messages.append(report)
    
    full_text = "\n\n---\n\n".join(final_messages)
    # 分段發送，WhatsApp 每則訊息建議不超過 1500 字元
    for i in range(0, len(full_text), 1500):
        send_whatsapp_message(full_text[i:i+1500])

if __name__ == "__main__":
    main()
