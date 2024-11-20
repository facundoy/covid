import covidcast
from datetime import date
from epiweeks import Week, Year
import pandas as pd
import os
# import pdb

## BELOW IS THE EXAMPLE CODE - update your own api key
#################################################################
#################################################################

michigan_counties = [
    '26001', '26003', '26005', '26007', '26009', '26011', '26013', '26015',
    '26017', '26019', '26021', '26023', '26025', '26027', '26029', '26031',
    '26033', '26035', '26037', '26039', '26041', '26043', '26045', '26047',
    '26049', '26051', '26053', '26055', '26057', '26059', '26061', '26063',
    '26065', '26067', '26069', '26071', '26073', '26075', '26077', '26079',
    '26081', '26083', '26085', '26087', '26089', '26091', '26093', '26095',
    '26097', '26099', '26101', '26103', '26105', '26107', '26109', '26111',
    '26113', '26115', '26117', '26119', '26121', '26123', '26125', '26127',
    '26129', '26131', '26133', '26135', '26137', '26139', '26141', '26143',
    '26145', '26147', '26149', '26151', '26153', '26155', '26157', '26159',
    '26161', '26163', '26165'
]

symptoms = [
    "anosmia_smoothed_search",
    "chest_pain_smoothed_search",
    "cough_smoothed_search",
    "diarrhea_smoothed_search",
    "difficulty_breathing_smoothed_search",
    "eye_pain_smoothed_search",
    "fatigue_smoothed_search",
    "fever_smoothed_search",
    "headache_smoothed_search",
    "loss_of_smell_smoothed_search",
    "muscle_pain_smoothed_search",
    "nausea_smoothed_search",
    "sore_throat_smoothed_search",
    "vomiting_smoothed_search", 
]

# query covidcast API for data
# Note:
#   smoothed_adj_cli is https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/doctor-visits.html
#   wcli (since April 15) from https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/fb-survey.html
#   and many more if we use since 2020-09-08
covidcast.use_api_key("163cdbefcb55b")
start_date, end_date = date(2020, 1, 1), date(2024, 6, 1)

def process_daily_data():
    doctor = covidcast.signal(
        "doctor-visits",
        "smoothed_adj_cli",
        start_date,
        end_date,
        "county",
        geo_values=michigan_counties
    )
    fb_wili = covidcast.signal(
        "fb-survey",
        "smoothed_wili",
        start_date,
        end_date,
        "county",
        geo_values=michigan_counties
    )
    fb_wcli = covidcast.signal(
        "fb-survey",
        "smoothed_wcli",
        start_date,
        end_date,
        "county",
        geo_values=michigan_counties
    )
    deaths = covidcast.signal(
        "indicator-combination",
        "deaths_incidence_num",
        start_date,
        end_date,
        "county",
        geo_values=michigan_counties
    )
    cases = covidcast.signal(
        "indicator-combination",
        "confirmed_incidence_num",
        start_date,
        end_date,
        "county",
        geo_values=michigan_counties
    )



    # symptom_signals = []
    # for symptom in symptoms:
    #     symptom_signal = covidcast.signal(
    #         "google-symptoms",
    #         symptom,
    #         start_date,
    #         end_date,
    #         "county",
    #         geo_values=michigan_counties
    #     )
    #     symptom_signals.append(symptom_signal)

    # data = covidcast.aggregate_signals([cases, deaths, doctor, fb_wili, fb_wcli] + symptom_signals)
    data = covidcast.aggregate_signals([cases, deaths, doctor, fb_wili, fb_wcli])

    data = data.rename(
        columns={
            "indicator-combination_confirmed_incidence_num_0_value": "cases",
            "indicator-combination_deaths_incidence_num_1_value": "deaths",
            "doctor-visits_smoothed_adj_cli_2_value": "doctor_visits",
            "fb-survey_smoothed_wili_3_value": "wili",
            "fb-survey_smoothed_wcli_4_value": "wcli"
        })

    # for i, symptom in enumerate(symptoms, start=5):
    #     readable_name = symptom.replace("_smoothed_search", "").replace("_", " ")
    #     data = data.rename(columns={f"google-symptoms_{symptom}_{i}_value": readable_name})


    output_dir = "delphi_county_data"

    for fips_code in michigan_counties:
        # county_data = data[data['geo_value'] == fips_code][[
        #     "time_value", "geo_value", "cases", "deaths", "doctor_visits", "wili", "wcli"
        # ] + [symptom.replace("_smoothed_search", "").replace("_", " ") for symptom in symptoms]]

        county_data = data[data['geo_value'] == fips_code][[
            "time_value", "geo_value", "cases", "deaths", "doctor_visits", "wili", "wcli"
        ]]
        
        output_file = os.path.join(output_dir, f"{fips_code}_data.csv")
        county_data.to_csv(output_file, index=False)

    print(data[[
        "time_value", "geo_value", "cases", "deaths", "doctor_visits", "wili",
        "wcli"
    ]].head(20))
    # data.head()

def process_and_save_weekly_data():

    input_dir = "delphi_county_data"
    output_dir = "delphi_weekly_county_data"

    def process_weekly_data(file_path):
        df = pd.read_csv(file_path)

        df['time_value'] = pd.to_datetime(df['time_value'], format='%Y-%m-%d')

        df.set_index('time_value', inplace=True)

        weekly_df = df.resample('W').sum() 

        weekly_df.reset_index(inplace=True)

        output_file = os.path.join(output_dir, os.path.basename(file_path))
        weekly_df.to_csv(output_file, index=False)

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)

        if file_name.endswith(".csv"):
            process_weekly_data(file_path)

    print("Weekly data has been successfully processed and saved.")

process_daily_data()