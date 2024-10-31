import pandas as pd


data = pd.read_csv('county_data.csv', parse_dates=['Date'])
data['Week'] = data['Date'] - pd.to_timedelta(data['Date'].dt.weekday, unit='D')


for county, county_data in data.groupby('COUNTY'):
    weekly_data = county_data.groupby('Week')['Cases'].sum().reset_index()

    filename = f'county_data/{county}_data.csv'
    weekly_data[['Week', 'Cases']].to_csv(filename, index=False)