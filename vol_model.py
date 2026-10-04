
import numpy as np, pandas as pd, yfinance as yf

df = yf.download('SPY', start='2005-01-01', auto_adjust=True)
df.columns = df.columns.get_level_values(0)
df['ret'] = np.log(df['Close']).diff()
vix = yf.download('^VIX', start='2005-01-01', auto_adjust=True)
vix.columns = vix.columns.get_level_values(0)
df['vix'] = vix['Close'] / 100 / np.sqrt(252)
df['vix_chg'] = df['vix'].diff()
print(df.tail())

H = 5
df['target'] = df['ret'].rolling(H).std().shift(-H)
df['rv5'] = df['ret'].rolling(5).std()
df['rv10'] = df['ret'].rolling(10).std()
df['rv20'] = df['ret'].rolling(20).std()
df['abs_ret'] = df['ret'].abs()
df['vol_chg'] = np.log(df['Volume']).diff()

df = df.dropna()
feats = ['rv5', 'rv10', 'rv20', 'abs_ret', 'vol_chg', 'vix', 'vix_chg']

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

models = {
    'linear' : LinearRegression(),
    'rf' : RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1),
    'xgb': XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05),
}

first_test = int((df.index < '2015-01-01').sum())
block = 63
preds = {n: pd.Series(index=df.index, dtype=float) for n in models}

for s in range(first_test, len(df), block):
    e = min(s+block, len(df))
    train = df.iloc[: s - H]
    test = df.iloc[s:e]
    for name, m in models.items():
        m.fit(train[feats], train['target'])
        preds[name].iloc[s:e] = m.predict(test[feats])

from arch import arch_model

garch = pd.Series(index=df.index, dtype=float)
r = df['ret'] * 100

for s in range(first_test, len(df), block):
    e = min(s + block, len(df))
    am = arch_model(r, vol='GARCH', p=1, q=1, mean='Constant')
    res = am.fit(last_obs=df.index[s - 1], disp='off')
    f = res.forecast(horizon=H, start=df.index[s], reindex=False)
    garch.iloc[s:e] = np.sqrt(f.variance.iloc[: e - s].mean(axis=1)) / 100


from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

out = pd.DataFrame({'actual': df['target'], 'garch': garch, **preds}).iloc[first_test:]
out['naive'] = df['rv5']
for c in ['garch', 'linear', 'rf', 'xgb', 'naive']:
    rmse = np.sqrt(mean_squared_error(out['actual'], out[c]))
    mae = mean_absolute_error(out['actual'], out[c])
    print(f'{c:7s} RMSE={rmse:.5f} MAE={mae:.5f}')

out[['actual', 'garch', 'xgb']].plot(figsize=(12,4))
plt.savefig('forecast_plot.png', dpi=150)