import pandas as pd
import matplotlib.pyplot as plt 
import scipy
import scipy.stats
import statsmodels.api as sm
import numpy as np 
from ggplot import *


# read in data
turnstile_weather = pd.read_csv('turnstile_weather_v2.csv')

# create histogram to assess how data is distributed
with_rain = turnstile_weather[turnstile_weather['rain'] == 1]
without_rain = turnstile_weather[turnstile_weather['rain'] == 0]

print ggplot(with_rain, aes(x = 'ENTRIESn_hourly')) + geom_histogram(binwidth = 250) +  labs('Total Entries in Four Hours', 'Count') + ggtitle('Histogram of Total Entries When Raining') + scale_y_continuous(labels='comma') + xlim(0,15000)
print ggplot(without_rain, aes(x = 'ENTRIESn_hourly')) + geom_histogram(binwidth = 250) +  labs('Total Entries in Four Hours', 'Count') + ggtitle('Histogram of Total Entries When Not Raining') + scale_y_continuous(labels='comma') + xlim(0,15000)


# the data is not normal and hence a T-test cannot be used to analyse the data. 

with_rain_mean = with_rain['ENTRIESn_hourly'].mean()
without_rain_mean = without_rain['ENTRIESn_hourly'].mean()

U, p = scipy.stats.mannwhitneyu(with_rain['ENTRIESn_hourly'], without_rain['ENTRIESn_hourly'])


# as p is below the critical value (0.05), then there is a statistical difference between the samples

def linear_regression(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    return intercept, params


features = turnstile_weather[[ 'hour', 'rain', 'weekday']]
dummy_units = pd.get_dummies(turnstile_weather['UNIT'], prefix='unit')
features = features.join(dummy_units)

# Values
values = turnstile_weather['ENTRIESn_hourly']

# Perform linear regression
intercept, params = linear_regression(features, values)
print params
predictions = intercept + np.dot(features, params)

plt.figure()
(turnstile_weather['ENTRIESn_hourly'] - predictions).hist(bins = 50)
plt.show()

# calculate r-squared to assess how effective regression is at predicting variations
def compute_r_squared(data, predictions):
	difference_pred = ((data-predictions)**2).sum()
	difference_mean = ((data - data.mean())**2).sum()
	r_squared = 1 - difference_pred/difference_mean
	return r_squared

print compute_r_squared(turnstile_weather['ENTRIESn_hourly'], predictions)

# create visualisation for number of riders per day of week
sum_per_day = turnstile_weather.groupby('day_week')['ENTRIESn_hourly'].sum()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

spd_df = pd.DataFrame({'days':days, 'total':sum_per_day})
print ggplot(spd_df, aes('days', y = 'total')) + geom_bar(stat='identity') + labs('Day', 'Total') + ggtitle('Total Subway Entries by Day') + scale_y_continuous(labels='comma')



