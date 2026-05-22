"""
批量训练 DL 模型（MLP）
覆盖所有有模型的股票 + 新股票
"""
import sys, os
sys.path.insert(0, 'bot')
import json, time

from dl_strategy import DLStrategy

# 完整股票池
ALL_WATCHLIST = [
    "NVDA","MSFT","AMD","META","AAPL","AMZN","GOOGL","GOOG","TSLA","NFLX",
    "AVGO","CRM","ORCL","CSCO","ADBE","ACN","IBM","INTC","QCOM","TXN","MU","LRCX",
    "JPM","V","MA","BAC","WFC","GS","MS","BLK","AXP","C",
    "SCHW","SPGI","MCO","CME","ICE","USB","PNC","TFC","COF","ADP","PLTR",
    "LLY","UNH","JNJ","ABBV","MRK","PFE","ABT","TMO","DHR","BMY",
    "AMGN","GILD","VRTX","REGN","ISRG","MDT","SYK","ZTS","BSX","EW",
    "CAT","DE","HON","GE","RTX","LMT","NOC","BA","UPS","FDX",
    "XOM","CVX","COP","SLB","EOG","MPC","VLO","PSX","OXY","CTAS",
    "PG","KO","PEP","COST","WMT","HD","MCD","SBUX","NKE","DIS",
    "CMCSA","VZ","T","TMUS","CHTR","EA","TTWO","LEN","DRI","NEE",
    "EXC","AEP","ORLY","AZO","ROST","BKR","PCAR","EL",
]

# 已有模型的 ticker（跳过）
EXISTING_MODELS = {
    "AAPL","AMD","AMZN","COIN","DIS","GOOGL","JNJ","JPM","KO",
    "META","MSFT","NFLX","NVDA","PLTR","SOFI","TSLA","UNH","V","XOM"
}

to_train = [t for t in ALL_WATCHLIST if t not in EXISTING_MODELS]
print(f"已有模型: {len(EXISTING_MODELS)} 个")
print(f"待训练: {len(to_train)} 个: {to_train}")
print()

results = []
success = 0
skip = 0
error = 0

for i, ticker in enumerate(to_train, 1):
    print(f"[{i}/{len(to_train)}] 训练 {ticker}...", end=" ", flush=True)
    try:
        dl = DLStrategy(ticker, model_type='MLP')
        result = dl.train(epochs=60, lookback=250)
        if 'error' in result:
            print(f"SKIP ({result['error']})")
            skip += 1
        else:
            print(f"OK acc={result['best_val_acc']:.3f}")
            success += 1
            results.append({ticker: result['best_val_acc']})
    except Exception as e:
        print(f"ERROR {e}")
        error += 1
    time.sleep(0.5)  # 避免请求太快

print()
print(f"=== 批量训练完成 ===")
print(f"成功: {success}  |  跳过: {skip}  |  错误: {error}")
print(f"成功率: {success}/{success+skip+error}")

# 保存结果
with open('data/batch_train_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("结果已保存 data/batch_train_results.json")