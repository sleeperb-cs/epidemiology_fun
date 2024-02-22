import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import ssl
import certifi
import urlopen


# Load COVID-19 dataset
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
data = pd.read_csv(url)

# Prepare data
data = data.drop(['Province/State', 'Lat', 'Long'], axis=1)
data = data.melt(id_vars='Country/Region', var_name='date', value_name='confirmed_cases')
data['date'] = pd.to_datetime(data['date'])
data['new_cases'] = data.groupby('Country/Region')['confirmed_cases'].diff().fillna(0)

# Plot global daily new cases
data.groupby('date')['new_cases'].sum().plot()
plt.title('Global Daily New COVID-19 Cases')
plt.show()

# Plot new cases for top 5 countries
countries = data['Country/Region'].value_counts().nlargest(5).index
fig, axs = plt.subplots(nrows=5, sharex=True)
for i, country in enumerate(countries):
  data[data['Country/Region'] == country].set_index('date')['new_cases'].plot(ax=axs[i])
  axs[i].set_title(country)
fig.tight_layout()
plt.show()

# Linear regression on global data
global_data = data.groupby('date')['new_cases'].sum().reset_index()
X = np.arange(len(global_data)).reshape(-1, 1)
y = global_data['new_cases'].values
model = LinearRegression()
model.fit(X, y)

# Plot regression trend
x_future = np.arange(len(global_data), len(global_data) + 30).reshape(-1, 1)
y_pred = model.predict(x_future)
plt.plot(global_data['date'], y)
plt.plot(pd.date_range(global_data['date'].iloc[-1], periods=30), y_pred)
plt.title('Predicted Global Cases Trend')

# Save analysis before showing the plot
plt.savefig('covid_analysis.pdf')

plt.show()