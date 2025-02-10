import pandas as pd
import numpy as np
import os
import csv

google_2020_mob_datapath = "full_google_mob_data/2020_US_Region_Mobility_Report.csv"
google_2021_mob_datapath = "full_google_mob_data/2021_US_Region_Mobility_Report.csv"
google_2022_mob_datapath = "full_google_mob_data/2022_US_Region_Mobility_Report.csv"


google_2020_df = pd.read_csv(google_2020_mob_datapath)
google_2021_df = pd.read_csv(google_2021_mob_datapath)
google_2022_df = pd.read_csv(google_2022_mob_datapath)

FIRST_COUNTY_CODE = 26001

start_index_2020 = google_2020_df[google_2020_df['census_fips_code'] == FIRST_COUNTY_CODE].index[0]
start_index_2021 = google_2021_df[google_2021_df['census_fips_code'] == FIRST_COUNTY_CODE].index[0]
start_index_2022 = google_2022_df[google_2022_df['census_fips_code'] == FIRST_COUNTY_CODE].index[0]

# print(f"{index_2020}, {index_2021}, {index_2022}")

NEXT_STATE = "Minnesota"

end_index_2020 = google_2020_df[google_2020_df['sub_region_1'] == NEXT_STATE].index[0]
end_index_2021 = google_2021_df[google_2021_df['sub_region_1'] == NEXT_STATE].index[0]
end_index_2022 = google_2022_df[google_2022_df['sub_region_1'] == NEXT_STATE].index[0]

# print(f"{end_index_2020}, {end_index_2021}, {end_index_2022}")

# Slice the dataframes for MI County data only:
google_MI_2020_df = google_2020_df.iloc[start_index_2020:end_index_2020]
google_MI_2021_df = google_2021_df.iloc[start_index_2021:end_index_2021]
google_MI_2022_df = google_2022_df.iloc[start_index_2022:end_index_2022]

# Convert 'census_fips_code' to integers
google_MI_2020_df = google_MI_2020_df.copy()
google_MI_2021_df = google_MI_2021_df.copy()
google_MI_2022_df = google_MI_2022_df.copy()

google_MI_2020_df['census_fips_code'] = google_MI_2020_df['census_fips_code'].astype('Int64')
google_MI_2021_df['census_fips_code'] = google_MI_2021_df['census_fips_code'].astype('Int64')
google_MI_2022_df['census_fips_code'] = google_MI_2022_df['census_fips_code'].astype('Int64')



file_path_2020 = "MI_county_google_mob_data/2020_MI_County_Mobility_Report.csv"
file_path_2021 = "MI_county_google_mob_data/2021_MI_County_Mobility_Report.csv"
file_path_2022 = "MI_county_google_mob_data/2022_MI_County_Mobility_Report.csv"

google_MI_2020_df.to_csv(file_path_2020, index=False)
google_MI_2021_df.to_csv(file_path_2021, index=False)
google_MI_2022_df.to_csv(file_path_2022, index=False)

# # Make the date to index dictionary:
# # Initialize the dictionary
# county_to_date_to_index = {}

# # Path to the folder containing the CSV files
# folder_path = "MI_county_google_mob_data"

# # List of CSV files in chronological order
# csv_files = ["2020_MI_County_Mobility_Report.csv", "2021_MI_County_Mobility_Report.csv", "2022_MI_County_Mobility_Report.csv"]

# # Initialize the index counter
# current_index = 0

# # Iterate through the files in order
# for csv_file in csv_files:
#     file_path = os.path.join(folder_path, csv_file)

#     # Open and read the CSV file
#     with open(file_path, mode='r') as file:
#         reader = csv.DictReader(file)

#         # Iterate through each row in the file
#         for row in reader:
#             curr_county = int(row['census_fips_code'])
#             if curr_county not in county_to_date_to_index:
#                 county_to_date_to_index[curr_county] = {}

#             date_str = row['date']

#             # Add the date and index to the county in the dictionary
#             county_to_date_to_index[curr_county][date_str] = current_index

#             # Increment the index for the next date
#             current_index += 1


# # Print the first 20 entries in the dictionary
# for i, (date, index) in enumerate(date_to_index.items()):
#     print(f"{date}: {index}")
#     if i == 19:  # Stop after printing 20 entries
#         break

# num = county_to_date_to_index[26003]["2021-03-27"]
# print(f"test: {num}")

def get_MI_dataframes_dict():
    google_MI_2020_df.reset_index(drop=True, inplace=True)
    google_MI_2021_df.reset_index(drop=True, inplace=True)
    google_MI_2022_df.reset_index(drop=True, inplace=True)
    
    dict_MI_df = {
        2020: google_MI_2020_df,
        2021: google_MI_2021_df,
        2022: google_MI_2022_df
    }
    return dict_MI_df

# def get_date_to_index_dict():
#     return county_to_date_to_index
