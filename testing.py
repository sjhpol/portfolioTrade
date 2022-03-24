from PortfolioOverview import Portfolio, Asset
import requests, zipfile, io, os
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

# Get Fama-French 5 factor data
import requests, zipfile, io, os
url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(os.getcwd())

file_name = "F-F_Research_Data_5_Factors_2x3_daily.CSV"

df = pd.read_csv(file_name, skiprows=3)
df['Date'] = df['Unnamed: 0'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
df.set_index('Date', inplace=True)

btc = Asset('BTC-USD')
end_date = df['Date'].astype(str).iloc[-1]
btc_data = btc.historical_data('2010-01-01', end_date)
btc_data['Log close'] = btc_data['Close'].apply(lambda x: np.log(x))
btc_data['log return'] = btc_data['Log close'] - btc_data['Log close'].shift()

model = btc_data['log return'].dropna()
model = model.to_frame().join(df, how='left')
model.dropna(inplace=True)

y = model['log return'] - model['RF']
X = model[['Mkt-RF', 'SMB', 'HML', 'CMA', 'RMW']]
X = sm.add_constant(X)
ff_model = sm.OLS(y, X).fit(use_t=True)
robust_model = ff_model.get_robustcov_results()
robust_model.summary()
intercept, b1, b2, b3, b4, b5 = robust_model.params


def t_test(t_value: float) -> float:
    if t_value > 0:
        return (1 - norm.cdf(t_value))*2
    else:
        return (norm.cdf(t_value))*2


t = t_test(robust_model.tvalues[0])

