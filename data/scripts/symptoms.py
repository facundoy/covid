import os
from google.cloud import bigquery
import pandas as pd
from epiweeks import Week
import numpy as np

def querydata():
    """
        Generate Google Symptoms Data from GCP Big Query for Michigan, saving daily data by sub_region_2_code
    """
    # Client read from credentials.json
    credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    client = bigquery.Client()
    
    epiweek = Week.thisweek(system="CDC")
    year = int(str(epiweek)[:-2])
    week = int(str(epiweek)[-2:])
    startdate = "2020-01-01"
    enddate = str((Week(year, week)-1).enddate())
    
    # SQL statement to extract data for Michigan
    sqlmichigan = f"""
    SELECT
    date,
    country_region,
    sub_region_1,
    sub_region_2_code,
    symptom_Fever,
    symptom_Low_grade_fever,
    symptom_Cough,
    symptom_Sore_throat,
    symptom_Headache,
    symptom_Fatigue,
    symptom_Vomiting,
    symptom_Diarrhea,
    symptom_Shortness_of_breath,
    symptom_Chest_pain,
    symptom_Dizziness,
    symptom_Confusion,
    symptom_Generalized_tonic_clonic_seizure,
    symptom_Weakness
    FROM `bigquery-public-data.covid19_symptom_search.symptom_search_sub_region_2_daily`
    WHERE country_region = 'United States' AND sub_region_1 = 'Michigan' AND date >= '{startdate}' AND date <= '{enddate}'
    ORDER BY sub_region_2_code ASC, date ASC
    """

    # Execute the Michigan query
    query_job = client.query(sqlmichigan)
    df = query_job.to_dataframe()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        
        # Correcting Columns
        df = df.rename(columns={"sub_region_1": "region"})
        df['region'].fillna("Michigan", inplace=True)
        df['country_region'].fillna("United States", inplace=True)
        
        # Ensure the output directory exists
        output_dir = os.path.join(os.getcwd(), "symptoms_county_data")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process and save each sub_region_2_code as a separate file
        for sub_region_2_code, group in df.groupby('sub_region_2_code'):
            if pd.notna(sub_region_2_code):  # Skip entries with missing sub_region_2_code
                filename = f"{sub_region_2_code}_data.csv"
                filepath = os.path.join(output_dir, filename)
                group.to_csv(filepath, index=False)
        
    else:
        print("No data available for Michigan in the specified date range.")

def combine_county_data(delphi_dir, symptoms_dir, combined_dir, weekly_dir):
    delphi_files = [file for file in os.listdir(delphi_dir) if file.endswith("_data.csv")]
    symptoms_files = [file for file in os.listdir(symptoms_dir) if file.endswith("_data.csv")]

    for file_name in delphi_files:
        delphi_file_path = os.path.join(delphi_dir, file_name)
        symptoms_file_path = os.path.join(symptoms_dir, file_name)

        if file_name not in symptoms_files:
            symptoms_file_path = None

        delphi_df = pd.read_csv(delphi_file_path)
        delphi_df["time_value"] = pd.to_datetime(delphi_df["time_value"])

        if symptoms_file_path:
            symptoms_df = pd.read_csv(symptoms_file_path)
            symptoms_df["date"] = pd.to_datetime(symptoms_df["date"])
            combined_df = pd.merge(delphi_df, symptoms_df, left_on="time_value", right_on="date", how="left")
            combined_df.drop(columns=["date", "country_region", "region", "geo_value", "sub_region_2_code"], inplace=True)
        else:
            for col in ["symptom_Fever", "symptom_Low_grade_fever", "symptom_Cough", "symptom_Sore_throat",
                        "symptom_Headache", "symptom_Fatigue", "symptom_Vomiting", "symptom_Diarrhea",
                        "symptom_Shortness_of_breath", "symptom_Chest_pain", "symptom_Dizziness", "symptom_Confusion",
                        "symptom_Generalized_tonic_clonic_seizure", "symptom_Weakness"]:
                delphi_df[col] = None

            combined_df = delphi_df

        combined_df.rename(columns={"time_value": "date"}, inplace=True)
        combined_df.to_csv(os.path.join(combined_dir, file_name), index=False)

        combined_df["week"] = combined_df["date"].dt.isocalendar().week
        combined_df["year"] = combined_df["date"].dt.year

        agg_columns = ["cases", "deaths", "doctor_visits", "wili", "wcli", "symptom_Fever", "symptom_Low_grade_fever",
                       "symptom_Cough", "symptom_Sore_throat", "symptom_Headache", "symptom_Fatigue",
                       "symptom_Vomiting", "symptom_Diarrhea", "symptom_Shortness_of_breath", "symptom_Chest_pain",
                       "symptom_Dizziness", "symptom_Confusion", "symptom_Generalized_tonic_clonic_seizure",
                       "symptom_Weakness"]

        weekly_df = combined_df.groupby(["year", "week"]).agg({col: "sum" for col in agg_columns}).reset_index()

        weekly_df["date"] = pd.to_datetime(weekly_df["year"].astype(str) + '-' + weekly_df["week"].astype(str) + '-1', 
                                           format='%Y-%U-%w')
        weekly_df["date"] = weekly_df["date"].dt.strftime('%Y-%m-%d')

        for col in agg_columns:
            weekly_df[col] = weekly_df[col].where(weekly_df[col].notna(), None)

        weekly_df = weekly_df[["date", "year", "week"] + agg_columns]

        weekly_df.to_csv(os.path.join(weekly_dir, file_name), index=False)

def check_missing_dates(delphi_dir):
    delphi_files = [file for file in os.listdir(delphi_dir) if file.endswith("_data.csv")]

    for file_name in delphi_files:
        file_path = os.path.join(delphi_dir, file_name)
        df = pd.read_csv(file_path)
        df["time_value"] = pd.to_datetime(df["time_value"])

        date_range = pd.date_range(start=df["time_value"].min(), end=df["time_value"].max())
        missing_dates = date_range.difference(df["time_value"])

        if not missing_dates.empty:
            print(f"Missing dates in {file_name}:")
            for date in missing_dates:
                print(date.strftime("%Y-%m-%d"))
        else:
            print(f"No missing dates in {file_name}.")

def fill_missing_dates(delphi_dir):
    start_date = pd.to_datetime("2020-02-01")
    end_date = pd.to_datetime("2024-06-01")

    delphi_files = [file for file in os.listdir(delphi_dir) if file.endswith("_data.csv")]

    for file_name in delphi_files:
        delphi_file_path = os.path.join(delphi_dir, file_name)
        delphi_df = pd.read_csv(delphi_file_path)
        delphi_df["time_value"] = pd.to_datetime(delphi_df["time_value"])
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_dates_df = pd.DataFrame(all_dates, columns=["time_value"])
        combined_df = pd.merge(all_dates_df, delphi_df, on="time_value", how="left")

        combined_df.to_csv(delphi_file_path, index=False)


delphi_dir = "delphi_county_data"
symptoms_dir = "symptoms_county_data"
combined_dir = "combined_county_data"
weekly_dir = "combined_weekly_county_data"

combine_county_data(delphi_dir, symptoms_dir, combined_dir, weekly_dir)
# check_missing_dates(delphi_dir)
# fill_missing_dates(delphi_dir)
# querydata()
