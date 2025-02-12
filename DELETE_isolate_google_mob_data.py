import pandas as pd
import numpy as np
import os

google_2020_mob_datapath = "MI_county_google_mob_data/2020_MI_County_Mobility_Report.csv"
google_2021_mob_datapath = "MI_county_google_mob_data/2021_MI_County_Mobility_Report.csv"
google_2022_mob_datapath = "MI_county_google_mob_data/2022_MI_County_Mobility_Report.csv"


google_2020_df = pd.read_csv(google_2020_mob_datapath)
google_2021_df = pd.read_csv(google_2021_mob_datapath)
google_2022_df = pd.read_csv(google_2022_mob_datapath)

T = 10

# List of Michigan county codes
# michigan_county_codes = [
#     "001", "003", "005", "007", "009", "011", "013", "015", "017", "019",
#     "021", "023", "025", "027", "029", "031", "033", "035", "037", "039",
#     "041", "043", "045", "047", "049", "051", "053", "055", "057", "059",
#     "061", "063", "065", "067", "069", "071", "073", "075", "077", "079",
#     "081", "083", "085", "087", "089", "091", "093", "095", "097", "099",
#     "101", "103", "105", "107", "109", "111", "113", "115", "117", "119",
#     "121", "123", "125", "127", "129", "131", "133", "135", "137", "139",
#     "141", "143", "145", "147", "149", "151", "153", "155", "157", "159",
#     "161", "163", "165"
# ]
michigan_county_codes = [
    "001", "003", "005", "007", "009", "011", "013", "015"
]

# Create a list of Michigan FIPS codes
michigan_fips_codes = ["26" + code for code in michigan_county_codes]

workdata_2020 = {}
workdata_2021 = {}
workdata_2022 = {}
resdata_2020 = {}
resdata_2021 = {}
resdata_2022 = {}

for micode in michigan_fips_codes:
    workdata_2020[micode] = []
    workdata_2021[micode] = []
    workdata_2022[micode] = []
    resdata_2020[micode] = []
    resdata_2021[micode] = []
    resdata_2022[micode] = []

    codeint = int(micode)
    index_2020 = google_2020_df[google_2020_df['census_fips_code'] == codeint].index[0]
    index_2021 = google_2021_df[google_2021_df['census_fips_code'] == codeint].index[0]
    index_2022 = google_2022_df[google_2022_df['census_fips_code'] == codeint].index[0]

    for i in range(T):
        work_value_2020 = google_2020_df.at[index_2020, 'workplaces_percent_change_from_baseline']
        res_value_2020 = google_2020_df.at[index_2020, 'residential_percent_change_from_baseline']
        work_value_2021 = google_2021_df.at[index_2021, 'workplaces_percent_change_from_baseline']
        res_value_2021 = google_2021_df.at[index_2021, 'residential_percent_change_from_baseline']
        work_value_2022 = google_2022_df.at[index_2022, 'workplaces_percent_change_from_baseline']
        res_value_2022 = google_2022_df.at[index_2022, 'residential_percent_change_from_baseline']
        if pd.isna(work_value_2020):
            work_value_2020 = 0.0
        if pd.isna(res_value_2020):
            res_value_2020 = 0.0
        if pd.isna(work_value_2021):
            work_value_2021 = 0.0
        if pd.isna(res_value_2021):
            res_value_2021 = 0.0
        if pd.isna(work_value_2022):
            work_value_2022 = 0.0
        if pd.isna(res_value_2022):
            res_value_2022 = 0.0

        workdata_2020[micode].append(work_value_2020)
        resdata_2020[micode].append(res_value_2020)
        workdata_2021[micode].append(work_value_2021)
        resdata_2021[micode].append(res_value_2021)
        workdata_2022[micode].append(work_value_2022)
        resdata_2022[micode].append(res_value_2022)

        index_2020 += 1
        index_2021 += 1
        index_2022 += 1

print(workdata_2020)
print()
print(resdata_2020)
print()
print(workdata_2021)
print()
print(resdata_2021)
print()
print(workdata_2022)
print()
print(resdata_2022)

def get_google_mob_data():
    return workdata_2020, resdata_2020, workdata_2021, resdata_2021, workdata_2022, resdata_2022
