import yfinance as yf
import requests
import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import urllib.parse
import sys
import json

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

DEFAULT_WEIGHTS = {"tech": 0.4, "news": 0.4, "fund": 0.2}

def get_today_events():
    return EVENTS.get(datetime.date.today().isoformat(), [])

def load_weights():
    try:
        with open("weights.json", "r") as f:
            return json.load(f)
    except:
        return DEFAULT_WEIGHTS

# ----------------------------- 工具函數 -----------------------------
def get_pivot_traditional(ticker):
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if len(data) < 2:
            return None, None, None
        prev = data.iloc[-2]
        h, l, c = float(prev['High']), float(prev['Low']), float(prev['Close'])
        pp = (h + l + c) / 3
        rng = h - l

        r1 = 2 * pp - l
        s1 = 2 * pp - h
        r2 = pp + rng
        s2 = pp - rng
        r3 = 2 * pp + (h - 2 * l)
        s3 = 2 * pp - (2 * h - l)

        signal = 1 if c > pp else (-1 if c < pp else 0)

        return {
            'pp': pp,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3,
            'close': c, 'signal': signal,
            'high': h, 'low': l, 'range': rng
        }, signal, c
    except Exception as e:
        print(f"Pivot錯誤 {ticker}: {e}")
        return None, None, None

def get_premarket_change(ticker):
    try:
        stock = yf.Ticker(ticker)
        pre = stock.info.get('preMarketPrice')
        prev_close = stock.info.get('regularMarketPreviousClose')
        if pre and prev_close:
            change_pct = (pre - prev_close) / prev_close * 100
            return pre, change_pct
    except:
        pass
    return None, None

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
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe = info.get('forwardPE') or info.get('trailingPE')
        pb = info.get('priceToBook')
        div_yield = info.get('dividendYield')
        if ticker == "QQQ":
            pe_low, pe_high = 25, 35
            pb_high = 8
        else:
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
        if div_yield and 0.03 < div_yield < 0.15:
            details.append(f"息率:{div_yield*100:.1f}%")
            signals.append(1)
        elif div_yield and div_yield >= 0.15:
            details.append(f"息率異常({div_yield*100:.1f}%)，已忽略")
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
    except:
        return 0, "數據缺失"

def send_whatsapp(text):
    encoded = urllib.parse.quote(text)
    url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={encoded}&apikey={WHATSAPP_API_KEY}"
    requests.get(url, timeout=10)

def send_report_safe(report, max_chars=1400):
    if len(report) <= max_chars:
        send_whatsapp(report)
    else:
        for i in range(0, len(report), max_chars):
            send_whatsapp(report[i:i+max_chars])

# ----------------------------- 昨日表現回顧與權重微調 -----------------------------
def get_yesterday_actual_direction(ticker):
    """回傳昨日實際方向：1(升), -1(跌), 0(平)"""
    try:
        data = yf.download(ticker, period="3d", progress=False)
        if len(data) < 3:
            return None
        day_before = data.iloc[-3]
        yesterday = data.iloc[-2]
        close_before = float(day_before['Close'])
        close_yesterday = float(yesterday['Close'])
        change = (close_yesterday - close_before) / close_before
        if change > 0.001:
            return 1
        elif change < -0.001:
            return -1
        else:
            return 0
    except:
        return None

def adjust_weights_and_threshold(ticker, base_weights):
    """根據昨日預測與實際對比，回傳 (adjusted_weights, adjusted_threshold)"""
    history_file = f"history_{TRADE_MAP[ticker]}.json"
    yesterday_str = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    try:
        with open(history_file, "r") as f:
            history = json.load(f)
    except:
        return base_weights, 0.4   # 無歷史記錄，回傳預設值

    # 尋找昨天的預測記錄
    yesterday_entry = None
    for entry in reversed(history):
        if entry.get("date") == yesterday_str:
            yesterday_entry = entry
            break
    if not yesterday_entry:
        return base_weights, 0.4

    # 取得昨日實際方向
    actual_dir = get_yesterday_actual_direction(ticker)
    if actual_dir is None:
        return base_weights, 0.4

    # 計算各組件的預測方向（用 >0.15 / <-0.15 離散化）
    def sig_to_dir(val):
        return 1 if val > 0.15 else (-1 if val < -0.15 else 0)

    tech_dir = sig_to_dir(yesterday_entry.get("tech_signal", 0))
    news_dir = sig_to_dir(yesterday_entry.get("news_signal", 0))
    fund_dir = sig_to_dir(yesterday_entry.get("fund_signal", 0))
    final_score = yesterday_entry.get("final_score", 0)
    overall_dir = 1 if final_score > 0.4 else (-1 if final_score < -0.4 else 0)

    # 動態調整各組件權重
    adj = base_weights.copy()
    # 技術
    if tech_dir == actual_dir:
        adj["tech"] *= 1.1
    else:
        adj["tech"] *= 0.8
    # 新聞
    if news_dir == actual_dir:
        adj["news"] *= 1.1
    else:
        adj["news"] *= 0.8
    # 基本面
    if fund_dir == actual_dir:
        adj["fund"] *= 1.1
    else:
        adj["fund"] *= 0.8

    # 歸一化
    total = sum(adj.values())
    for k in adj:
        adj[k] /= total

    # 門檻調整：若整體預測錯誤，提高門檻
    new_threshold = 0.4
    if overall_dir != actual_dir and overall_dir != 0:
        new_threshold = 0.55

    return adj, new_threshold

# ----------------------------- 報告產生 -----------------------------
def build_report(name, ticker, trade_inst, pivot_data, news_avg, top3_news, events,
                 fund_signal, fund_text, pre_price, pre_chg, weights, threshold):
    today = datetime.date.today()
    if not pivot_data:
        return None, None

    p = pivot_data
    price = p['close']
    pp = p['pp']
    r1, r2, r3 = p['r1'], p['r2'], p['r3']
    s1, s2, s3 = p['s1'], p['s2'], p['s3']
    rng = p['range']

    vol_pct = (rng / price) * 100
    is_low_vol = vol_pct < 1.0

    news_signal = 1 if news_avg > 0.15 else (-1 if news_avg < -0.15 else 0)

    # 新聞極度中性時仍動態調權（與昨日調整疊加）
    adj_weights = weights.copy()
    if abs(news_avg) <= 0.1:
        adj_weights["tech"] = min(0.6, adj_weights["tech"] + 0.1)
        adj_weights["news"] = max(0.2, adj_weights["news"] - 0.1)
        total = sum(adj_weights.values())
        for k in adj_weights:
            adj_weights[k] /= total

    pre_bonus = 0
    pre_note = ""
    if pre_price:
        pre_note = f"盤前: {pre_price:.2f} ({pre_chg:+.2f}%)"
        if pre_price > r1:
            pre_bonus = 0.3
            pre_note += " 已破R1"
        elif pre_price > pp:
            pre_bonus = 0.1
        elif pre_price < s1:
            pre_bonus = -0.3
            pre_note += " 已破S1"
        elif pre_price < pp:
            pre_bonus = -0.1

    final_score = (adj_weights["tech"] * (p['signal'] + pre_bonus) +
                   adj_weights["news"] * news_signal +
                   adj_weights["fund"] * fund_signal)

    event_str = ""
    if events:
        event_str = "⚠️ 今日事件: " + ", ".join(events) + "\n"

    news_lines = []
    for title, score, weight in top3_news:
        emoji = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
        news_lines.append(f"{emoji} [{weight}x] {title[:80]}")

    # 最終門檻：低波動時再稍微提高
    final_threshold = threshold
    if is_low_vol:
        final_threshold = max(final_threshold, 0.5)

    if final_score > final_threshold:
        prediction = "📈 上升 (做多)"
        entry = r1
        target = r2
        stop = pp
        reward = target - entry
        risk = entry - stop
        if reward / risk < 1.5:
            plan = (f"⚠️ 盈虧比不佳 ({reward/risk:.1f})，建議觀望\n"
                    f"若仍想操作：突破 R1 {r1:.2f} 後買入看漲，目標 R2 {r2:.2f}，止損 PP {pp:.2f}")
        else:
            plan = (f"入場：突破 R1 {r1:.2f} 後買入看漲期權\n"
                    f"目標：R2 {r2:.2f}，延伸 R3 {r3:.2f}\n"
                    f"止損：跌破 PP {pp:.2f} 或期權價值減半\n"
                    f"⚙️ 動態管理：若達 R1+0.3*Range ({r1+0.3*rng:.2f})，止損上移至入場價")
    elif final_score < -final_threshold:
        prediction = "📉 下跌 (做空)"
        entry = s1
        target = s2
        stop = pp
        reward = entry - target
        risk = stop - entry
        if reward / risk < 1.5:
            plan = (f"⚠️ 盈虧比不佳 ({reward/risk:.1f})，建議觀望\n"
                    f"若仍想操作：跌破 S1 {s1:.2f} 後買入看跌，目標 S2 {s2:.2f}，止損 PP {pp:.2f}")
        else:
            plan = (f"入場：跌破 S1 {s1:.2f} 後買入看跌期權\n"
                    f"目標：S2 {s2:.2f}，延伸 S3 {s3:.2f}\n"
                    f"止損：升破 PP {pp:.2f} 或期權價值減半\n"
                    f"⚙️ 動態管理：若達 S1-0.3*Range ({s1-0.3*rng:.2f})，止損下移至入場價")
    else:
        prediction = "↔️ 震盪 (區間交易)"
        plan = (f"操作：於 S1 {s1:.2f} 附近買入看漲，目標 R1 {r1:.2f}\n"
                f"　　　或於 R1 {r1:.2f} 附近買入看跌，目標 S1 {s1:.2f}\n"
                f"止損：突破 S2 {s2:.2f} 或 R2 {r2:.2f}")

    ext_levels = (
        f"R1:{r1:.1f} R2:{r2:.1f} R3:{r3:.1f}\n"
        f"S1:{s1:.1f} S2:{s2:.1f} S3:{s3:.1f}"
    )

    report = (
        f"📅 {today.isoformat()} | {name} ({trade_inst})\n"
        f"{event_str}"
        f"💰 前收: {price:.2f}  樞軸(Trad): {pp:.2f}  日波幅: {vol_pct:.1f}%\n"
        f"{pre_note + chr(10) if pre_note else ''}"
        f"📊 基本面: {fund_text}\n"
        f"🗞️ 新聞情緒: {news_avg:.2f}\n"
        f"⚖️ 動態權重: T{adj_weights['tech']:.2f} N{adj_weights['news']:.2f} F{adj_weights['fund']:.2f} | 門檻:{final_threshold:.2f}\n"
        f"--- 關鍵新聞 ---\n" + "\n".join(news_lines) + "\n"
        f"🎯 預測: {prediction}\n"
        f"{plan}\n"
        f"📐 關鍵水平:\n{ext_levels}\n"
        f"⚡ 0DTE警告：嚴格止損，時間價值損耗快，僅限極短線。"
    )

    signals = {
        "tech_signal": p['signal'],
        "news_signal": news_signal,
        "fund_signal": fund_signal,
        "final_score": final_score
    }
    return report, signals

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
    since = (today - datetime.timedelta(days=1)).isoformat()
    events = get_today_events()
    base_weights = load_weights()

    for name, ticker in TICKERS.items():
        if target and ticker != target:
            continue

        # 1. 取得昨日預測vs實際，即時調整權重與門檻
        adj_weights, threshold = adjust_weights_and_threshold(ticker, base_weights)

        # 2. 樞軸點計算
        pivot_data, _, _ = get_pivot_traditional(ticker)
        if not pivot_data:
            send_whatsapp(f"⚠️ {name} 數據缺失")
            continue

        # 3. 盤前價格
        pre_price, pre_chg = get_premarket_change(ticker)

        # 4. 新聞
        query = NEWS_QUERIES.get(ticker, name)
        titles = fetch_news(query, since, today.isoformat())
        if not titles:
            titles = ["無相關新聞"]
        news_avg, _, top3 = analyze_news(titles)

        # 5. 基本面
        fund_signal, fund_text = get_fundamental_signal(ticker)

        # 6. 生成報告
        report, signals = build_report(name, ticker, TRADE_MAP.get(ticker, name),
                                       pivot_data, news_avg, top3, events,
                                       fund_signal, fund_text, pre_price, pre_chg,
                                       adj_weights, threshold)
        if report:
            send_report_safe(report)

            # 寫入歷史
            history_file = f"history_{TRADE_MAP[ticker]}.json"
            entry = {
                "date": today.isoformat(),
                "ticker": ticker,
                "tech_signal": signals["tech_signal"],
                "news_signal": signals["news_signal"],
                "fund_signal": signals["fund_signal"],
                "final_score": signals["final_score"],
                "weights": adj_weights
            }
            try:
                with open(history_file, "r") as f:
                    hist = json.load(f)
            except:
                hist = []
            hist.append(entry)
            with open(history_file, "w") as f:
                json.dump(hist, f, indent=2)

if __name__ == "__main__":
    main()
