import warnings
import itertools
import pandas as pd
import statsmodels.api as sm


'''
This file creates a CSV of the forecasts resulting from all permutations of
the SARIMAX parameters
'''

df = pd.read_csv('AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])

# Set up the SARIMAX model
y = pd.Series(data=df['#Passengers'].values, index=df['Month'])
model = pd.DataFrame()

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q, and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q, and q triplets
seasonal_pdq = list(itertools.product(p, d, q, [1, 3, 6, 9, 12]))

warnings.filterwarnings("ignore")   # specify to ignore warning messages

# Grid search all p, d, q, P, D, Q permutations and save resulting forecast
best_result = [0, 0, 10000000]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        # Try/Except in order to pass by misspecifications
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit(disp=False)
            pred = results.get_prediction(start=pd.to_datetime('1949-01-01'),
                                          dynamic=False)
            pred_uc = results.get_forecast(steps=96)

            model_data = list(pred.predicted_mean)
            model_data.extend(list(pred_uc.predicted_mean))
            model['ARIMA{} x {}'.format(param, param_seasonal)] = model_data

            print('ARIMA{} x {} - AIC: {}'.format(param,
                                                  param_seasonal,
                                                  results.aic))

            if results.aic < best_result[2]:
                best_result = [param, param_seasonal, results.aic]
        except Exception:
            continue

print('\nBest Result:', best_result)

# Add actual data to DataFrame
model['AirPassengers'] = df['#Passengers']

# Create index from corresponding dates
model['Date'] = [pd.to_datetime('{}-{}'.format(year, month))
                 for year in range(1949, 1969) for month in range(1, 13)]
model = model.set_index('Date')

model.to_csv('arima_permutations.csv')
