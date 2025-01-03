import pandas as pd


data = pd.read_csv('county_data.csv', parse_dates=['Date'])
data['Week'] = data['Date'] - pd.to_timedelta(data['Date'].dt.weekday, unit='D')

fips_mapping = {
    'Alcona': '26001', 'Alger': '26003', 'Allegan': '26005', 'Alpena': '26007', 'Antrim': '26009',
    'Arenac': '26011', 'Baraga': '26013', 'Barry': '26015', 'Bay': '26017', 'Benzie': '26019',
    'Berrien': '26021', 'Branch': '26023', 'Calhoun': '26025', 'Cass': '26027', 'Charlevoix': '26029',
    'Cheboygan': '26031', 'Chippewa': '26033', 'Clare': '26035', 'Clinton': '26037', 'Crawford': '26039',
    'Delta': '26041', 'Dickinson': '26043', 'Eaton': '26045', 'Emmet': '26047', 'Genesee': '26049',
    'Gladwin': '26051', 'Gogebic': '26053', 'Grand Traverse': '26055', 'Gratiot': '26057',
    'Hillsdale': '26059', 'Houghton': '26061', 'Huron': '26063', 'Ingham': '26065', 'Ionia': '26067',
    'Iosco': '26069', 'Iron': '26071', 'Isabella': '26073', 'Jackson': '26075', 'Kalamazoo': '26077',
    'Kalkaska': '26079', 'Kent': '26081', 'Keweenaw': '26083', 'Lake': '26085', 'Lapeer': '26087',
    'Leelanau': '26089', 'Lenawee': '26091', 'Livingston': '26093', 'Luce': '26095', 'Mackinac': '26097',
    'Macomb': '26099', 'Manistee': '26101', 'Marquette': '26103', 'Mason': '26105', 'Mecosta': '26107',
    'Menominee': '26109', 'Midland': '26111', 'Missaukee': '26113', 'Monroe': '26115', 'Montcalm': '26117',
    'Montmorency': '26119', 'Muskegon': '26121', 'Newaygo': '26123', 'Oakland': '26125', 'Oceana': '26127',
    'Ogemaw': '26129', 'Ontonagon': '26131', 'Osceola': '26133', 'Oscoda': '26135', 'Otsego': '26137',
    'Ottawa': '26139', 'Presque Isle': '26141', 'Roscommon': '26143', 'Saginaw': '26145', 'St. Clair': '26147',
    'St. Joseph': '26149', 'Sanilac': '26151', 'Schoolcraft': '26153', 'Shiawassee': '26155',
    'Tuscola': '26157', 'Van Buren': '26159', 'Washtenaw': '26161', 'Wayne': '26163', 'Wexford': '26165'
}


for county, county_data in data.groupby('COUNTY'):
    fips_code = fips_mapping.get(county)
    if fips_code is None:
        print(f"Warning: FIPS code not found for county '{county}'")
        continue
    weekly_data = county_data.groupby('Week')['Cases'].sum().reset_index()

    filename = f'county_data/{fips_code}_data.csv'
    weekly_data[['Week', 'Cases']].to_csv(filename, index=False)