import yfinance as yf
import json
import datetime

TICKERS = {
    "QQQ": "QQQ",
    "騰訊700": "0700.HK"
}
TRADE_MAP = {"QQQ": "QQQ", "0700.HK": "騰訊700"}

# 預設權重，若無法計算則回退
DEFAULT_WEIGHTS = {"tech": 0.4, "news": 0.4, "fund": 0.2}

def get_actual_direction(ticker):
    """比較今日與昨日收盤價，回傳實際方向：1(升), -1(跌), 0(持平)"""
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if len(data) < 2:
            return None
        yesterday = data.iloc[-2]
        today = data.iloc[-1]
        close_yest = float(yesterday['Close'])
        close_today = float(today['Close'])
        change = (close_today - close_yest) / close_yest
        if change > 0.001:
            return 1
        elif change < -0.001:
            return -1
        else:
            return 0
    except Exception as e:
        print(f"無法取得 {ticker} 實際收盤：{e}")
        return None

def main():
    for ticker, name in TICKERS.items():
        history_file = f"history_{TRADE_MAP[ticker]}.json"
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except:
            print(f"無歷史檔案 {history_file}，跳過")
            continue

        # 找最後一筆未填 actual 的記錄
        updated = False
        for entry in reversed(history):
            if "actual" not in entry:
                actual = get_actual_direction(ticker)
                if actual is not None:
                    entry["actual"] = actual
                    # 判斷預測方向是否正確
                    pred_dir = 1 if entry["final_score"] > 0.4 else (-1 if entry["final_score"] < -0.4 else 0)
                    entry["match"] = 1 if pred_dir == actual else 0
                    updated = True
                break

        if not updated:
            print(f"{name} 沒有需要更新的預測記錄")
            continue

        # 儲存更新後的歷史
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        # 根據最近20筆計算各組件準確率
        valid = [h for h in history if "actual" in h and "match" in h][-20:]
        if len(valid) < 5:
            print(f"{name} 有效記錄不足5筆，維持原權重")
            continue

        # 各組件單獨預測方向 (用 > 0.15 / < -0.15 閾值)
        def signal_to_dir(sig):
            return 1 if sig > 0.15 else (-1 if sig < -0.15 else 0)

        tech_acc = sum(1 for h in valid if signal_to_dir(h["tech_signal"]) == h["actual"]) / len(valid)
        news_acc = sum(1 for h in valid if signal_to_dir(h["news_signal"]) == h["actual"]) / len(valid)
        fund_acc = sum(1 for h in valid if signal_to_dir(h["fund_signal"]) == h["actual"]) / len(valid)

        total = tech_acc + news_acc + fund_acc
        if total == 0:
            continue

        new_weights = {
            "tech": tech_acc / total,
            "news": news_acc / total,
            "fund": fund_acc / total
        }
        with open("weights.json", "w") as f:
            json.dump(new_weights, f, indent=2)
        print(f"✅ {name} 權重已更新：{new_weights}")

if __name__ == "__main__":
    main()
