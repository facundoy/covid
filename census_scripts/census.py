"""From census API, obtain multiple distributions for a county
    - household size distribution
    - age distribution
    - occupation distribution
    Variables as per https://api.census.gov/data/2021/acs/acs5/variables.html

    TODO: add SES, race from mortage data
"""

import requests
import pandas as pd
import pdb
import json
import os
import wget
import sys
from tqdm import tqdm
import numpy as np


def obtain_household_size_distribution(county_fips, census_api_key):
    """ Obtain household size distribution for county from Census data"""

    # Define the variables you want to query
    household_variables = [
        "B11016_001E", "B11016_003E", "B11016_011E", "B11016_004E",
        "B11016_012E", "B11016_005E", "B11016_013E", "B11016_006E",
        "B11016_014E", "B11016_007E", "B11016_008E", "B11016_015E",
        "B11016_016E"
    ]

    # Define the dictionary with the column mapping
    household_size_acs_var_map = {
        "1":
        "B11016_001E - B11016_003E - B11016_011E - B11016_004E -"
        "B11016_012E - B11016_005E - B11016_013E - B11016_006E -"
        "B11016_014E - B11016_007E - B11016_008E - B11016_015E -"
        "B11016_016E",
        "2":
        "B11016_003E + B11016_011E",
        "3":
        "B11016_004E + B11016_012E",
        "4":
        "B11016_005E + B11016_013E",
        "5":
        "B11016_006E + B11016_014E",
        "6":
        "B11016_007E + B11016_008E + B11016_015E + B11016_016E",
    }

    base_url = "https://api.census.gov/data/2020/acs/acs5"
    variables_str = ",".join(household_variables)
    url = f"{base_url}?get=NAME,{variables_str}&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"

    response = requests.get(url)

    # Check the response status code
    if response.status_code == 200:
        # The request was successful, so parse the JSON response
        results = json.loads(response.content)
        # construct dataframe
        df = pd.DataFrame(results[1:], columns=results[0])
        # convert all columns to numeric
        df = df.apply(pd.to_numeric, errors='ignore')
        # Create new columns based on the dictionary mapping
        print(f"Obtaining Household Data")
        for new_column, formula in household_size_acs_var_map.items():
            # print(f"{new_column}/6")
            df[new_column] = df.eval(formula)
    else:
        # The request failed, so print the error message
        print(response.status_code, response.reason)
        print('Failed for county: ', county_fips)
        quit()

    # keep only new columns
    df = df[list(household_size_acs_var_map.keys())]
    """ Save with format as csv file
        Household Size,Number
        1,329114
        2,306979
        3,139176
        4,115757
        5,45162
        6,30375
    """
    # transpose dataframe
    df = df.transpose()
    # reset index
    df = df.reset_index()
    # rename columns
    df = df.rename(columns={"index": "Household Size", 0: "Number"})
    df.to_csv(f"./data/{county_fips}/agents_household_sizes.csv", index=False)


def obtain_age_distribution(county_fips, census_api_key):
    """ Obtain age distribution for county from Census data """

    age_variables = [
        "B01001_003E", "B01001_004E", "B01001_005E", "B01001_006E",
        "B01001_007E", "B01001_027E", "B01001_028E", "B01001_029E",
        "B01001_030E", "B01001_031E", "B01001_008E", "B01001_009E",
        "B01001_010E", "B01001_011E", "B01001_012E", "B01001_032E",
        "B01001_033E", "B01001_034E", "B01001_035E", "B01001_036E",
        "B01001_013E", "B01001_014E", "B01001_015E", "B01001_016E",
        "B01001_017E", "B01001_037E", "B01001_038E", "B01001_039E",
        "B01001_040E", "B01001_041E", "B01001_018E", "B01001_019E",
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E",
        "B01001_024E", "B01001_025E", "B01001_042E", "B01001_043E",
        "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E",
        "B01001_048E", "B01001_049E"
    ]

    # Define the dictionary with the column mapping
    age_acs_var_map = {
        "AGE_0_9":
        "B01001_003E + B01001_004E + B01001_027E + B01001_028E",
        "AGE_10_19":
        "B01001_005E + B01001_006E + B01001_007E + B01001_029E +"
        "B01001_030E + B01001_031E",
        "AGE_20_29":
        "B01001_008E + B01001_009E + B01001_010E + "
        "B01001_011E + B01001_032E + B01001_033E + "
        "B01001_034E + B01001_035E",
        "AGE_30_39":
        "B01001_012E + B01001_013E + B01001_036E + B01001_037E",
        "AGE_40_49":
        "B01001_014E + B01001_015E + B01001_038E + B01001_039E",
        "AGE_50_59":
        "B01001_016E + B01001_017E + B01001_040E + B01001_041E",
        "AGE_60_69":
        "B01001_018E + B01001_019E + B01001_020E + B01001_021E +"
        "B01001_042E + B01001_043E + B01001_044E + B01001_045E",
        "AGE_70_79":
        "B01001_022E + B01001_023E + B01001_046E + B01001_047E",
        "AGE_80":
        "B01001_024E + B01001_025E + B01001_048E + B01001_049E"
    }

    base_url = "https://api.census.gov/data/2020/acs/acs5"
    variables_str = ",".join(age_variables)
    url = f"{base_url}?get=NAME,{variables_str}&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"

    response = requests.get(url)

    # Check the response status code
    if response.status_code == 200:
        # The request was successful, so parse the JSON response
        results = json.loads(response.content)
        # construct dataframe
        df = pd.DataFrame(results[1:], columns=results[0])
        # convert all columns to numeric
        df = df.apply(pd.to_numeric, errors='ignore')
        # Create new columns based on the dictionary mapping
        # print('-------------------')
        print(f"Obtaining Age Data")
        for new_column, formula in age_acs_var_map.items():
            # print(f"{new_column}/6")
            df[new_column] = df.eval(formula)
    else:
        # The request failed, so print the error message
        print(response.status_code, response.reason)
        print('Failed for county: ', county_fips)
        quit()

    # keep only new columns
    df = df[list(age_acs_var_map.keys())]
    """ Save with format as csv file
        Age,Number
        AGE_0_9,278073
        AGE_10_19,258328
        AGE_20_29,317005
        AGE_30_39,359688
        AGE_40_49,323457
        AGE_50_59,307938
        AGE_60_69,229274
        AGE_70_79,109487
        AGE_80,69534
    """
    # transpose dataframe
    df = df.transpose()
    # reset index
    df = df.reset_index()
    # rename columns
    df = df.rename(columns={"index": "Age", 0: "Number"})
    df.to_csv(f"./data/{county_fips}/agents_ages.csv", index=False)


def obtain_occupation_distribution(county_fips, census_api_key):
    """ Obtain occupation distribution for county from Census data 
        Using NAICS sectors as Abueg et al. 2021, npj Digital Medicine
        
    """
    # List of NAICS sectors to query as per
    # https://www.census.gov/programs-surveys/economic-census/year/2022/guidance/understanding-naics.html
    # NOTE: only for reference, using inverted_naics_sectors instead
    naics_sectors = {
        "11": "Agriculture, Forestry, Fishing and Hunting",
        "21": "Mining, Quarrying, and Oil and Gas Extraction",
        "22": "Utilities",
        "23": "Construction",
        "31-33": "Manufacturing",
        "42": "Wholesale Trade",
        "44-45": "Retail Trade",
        "48-49": "Transportation and Warehousing",
        "51": "Information",
        "52": "Finance and Insurance",
        "53": "Real Estate and Rental and Leasing",
        "54": "Professional, Scientific, and Technical Services",
        "55": "Management of Companies and Enterprises",
        "56":
        "Administrative and Support and Waste Management and Remediation Services",
        "61": "Educational Services",
        "62": "Health Care and Social Assistance",
        "71": "Arts, Entertainment, and Recreation",
        "72": "Accommodation and Food Services",
        "81": "Other Services (except Public Administration)",
        "92": "Public Administration"
    }
    # Define the dictionary with the sector code mapping
    inverted_naics_sectors = {
        'AGRICULTURE': ['11'],
        'MINING': ['21'],
        'UTILITIES': ['22'],
        'CONSTRUCTION': ['23'],
        'MANUFACTURING': ['31', '32', '33'],
        'WHOLESALETRADE': ['42'],
        'RETAILTRADE': ['44', '45'],
        'TRANSPORTATION': ['48', '49'],
        'INFORMATION': ['51'],
        'FINANCEINSURANCE': ['52'],
        'REALESTATERENTAL': ['53'],
        'SCIENTIFICTECHNICAL': ['54'],
        'ENTERPRISEMANAGEMENT': ['55'],
        'WASTEMANAGEMENT': ['56'],
        'EDUCATION': ['61'],
        'HEALTHCARE': ['62'],
        'ART': ['71'],
        'FOOD': ['72'],
        'OTHER': ['81', '92'],  # adding public administration to other
    }

    df = [['Occupation', 'Number']]
    # convert example api.census.gov/data/2017/ecnbasic?get=NAICS2017_LABEL,EMP,NAME,GEO_ID&for=us:*&NAICS2017=54&key=YOUR_KEY_GOES_HERE
    base_url = "https://api.census.gov/data/2017/ecnbasic"
    # print('-------------------')
    print(f"Obtaining Occupation Data")
    for naics_sector in inverted_naics_sectors:
        # print(naics_sector)
        # print('-------------------')
        for naics_sector_code in inverted_naics_sectors[naics_sector]:
            # print(naics_sector, naics_sector_code)
            url = f"{base_url}?get=NAICS2017_LABEL,EMP,NAME,GEO_ID&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&NAICS2017={naics_sector_code}&key={census_api_key}"
            # Make the API request
            response = requests.get(url)
            # Check the response status code
            if response.status_code == 200:
                # The request was successful, so parse the JSON response
                results = json.loads(response.content)
                row = results[1][:2]
                # label sector
                row[0] = naics_sector
                df.append(row)
            else:
                # The request failed, so print the error message
                # print(response.status_code, response.reason)
                row = [naics_sector, 0]
                df.append(row)

    # construct dataframe
    df = pd.DataFrame(df[1:], columns=df[0])
    # convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='ignore')
    # aggregate by sector but do not sort by sector
    df = df.groupby(['Occupation'], sort=False).sum()
    # reset index
    df = df.reset_index()
    df.to_csv(f"./data/{county_fips}/agents_occupations.csv", index=False)


"""
    CSV files to be created:

    Step 1: start with this:
    
        county_fips, var1, var2, ..., var8
        27109, 0.1, 0.4, ..., 0.1 = 0.9
        27049, 0.2, 0.3, ..., 0.1 = 0.78
    
    Step 2: check there are variations across counties

    Step 3: then do this:

        county_fips, census_block, var1, var2, ..., var8
        27109, 001, 0.1, 0.4, ..., 0.1 = 0.9
        27109, 002, 0.2, 0.3, ..., 0.1 = 0.78

        >>> it gives you diversity across one county

    once done with MN, test in GA counties
"""
# by leo


def obtain_people_living_along(county_fips, census_api_key):
    base_url = "https://api.census.gov/data/2021/acs/acs5"
    url = f"{base_url}?get=NAME,group(B11001)&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"
    response = requests.get(url)
    results = json.loads(response.content)

    code_total_household = "B11001_001E"
    code_nonfamily_living_alone = "B11001_008E"

    df = pd.DataFrame(results[1:], columns=results[0])
    return float(df.loc[0, code_nonfamily_living_alone]) / float(
        df.loc[0, code_total_household])


def obtain_elder_living_along(county_fips, census_api_key):
    base_url = "https://api.census.gov/data/2011/acs/acs5"
    url = f"{base_url}?get=NAME,group(B09017)&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"
    response = requests.get(url)
    results = json.loads(response.content)

    code_total_household = "B09017_001E"
    code_nonfamily_male_living_alone = "B09017_014E"
    code_nonfamily_female_living_alone = "B09017_017E"

    df = pd.DataFrame(results[1:], columns=results[0])

    return (float(df.loc[0, code_nonfamily_male_living_alone]) + float(df.loc[0, code_nonfamily_female_living_alone])) \
        / float(df.loc[0, code_total_household])


def obtain_family_with_grandchild(county_fips, census_api_key):
    base_url = "https://api.census.gov/data/2021/acs/acs5"
    url = f"{base_url}/subject?get=NAME,group(S1002)&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"
    response = requests.get(url)
    results = json.loads(response.content)

    code_total_household = "S1002_C01_029E"
    code_grandparents_household = "S1002_C01_001E"

    df = pd.DataFrame(results[1:], columns=results[0])

    return float(df.loc[0, code_grandparents_household]) / float(
        df.loc[0, code_total_household])


def obtain_marry_divorce_ratio(county_fips, census_api_key):
    base_url = "https://api.census.gov/data/2021/acs/acs5"
    url = f"{base_url}/subject?get=NAME,group(S1201)&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"
    response = requests.get(url)
    results = json.loads(response.content)

    code_marriage = "S1201_C02_001E"
    code_divorce = "S1201_C04_001E"

    df = pd.DataFrame(results[1:], columns=results[0])

    return float(df.loc[0, code_marriage]) / float(df.loc[0, code_divorce])


def obtain_religious(county_fips):
    df = pd.read_excel('./data/2020_USRC_Summaries.xlsx', sheet_name=2)

    df = df[df['FIPS'] == county_fips]

    religious = df['Adherents']
    total = df['2020 Population']

    return float(total - religious) / float(total)


# TODO: Check file source


def obtain_self_employ(county_fips):
    # base_url = "https://api.census.gov/data/2021/acs/acsse"
    # url = f"{base_url}?get=NAME,K202402&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"
    # response = requests.get(url)
    # results = json.loads(response.content)

    df = pd.read_csv('./data/ACSSE2021.K202402-Data.csv',
                     encoding='latin-1',
                     skiprows=[1])
    geo_id = "0500000US" + county_fips
    df = df[df["GEO_ID"] == geo_id]

    code_total = "K202402_001E"
    code_self_own = "K202402_003E"
    code_self_nonown = "K202402_008E"

    return (float(df[code_self_own]) + float(df[code_self_nonown])) \
        / float(df[code_total])


# def obtain_voting(county_fips, state_dict, state_county_dict):
#     state = state_dict[county_fips[:2]]
#     if len(state.split()) >= 2:
#         state = state.split()
#         if len(state) == 3:
#             state = state[1] + '_' + state[2] + '_' + state[3]
#         elif len(state) == 2:
#             state = state[1] + '_' + state[2]
#     county_dict = state_county_dict[county_fips[:2]]
#     county = county_dict[county_fips]

#     target_frame = pd.MultiIndex.from_arrays([
#         ['Jo Jorgensen Libertarian', 'Jo Jorgensen Libertarian'],
#         ['%', '#']
#     ])

#     target_frame1 = pd.MultiIndex.from_arrays([
#         ['Jo Jorgensen Independent', 'Jo Jorgensen Independent'],
#         ['%', '#']
#     ])

#     # print(state)

#     url = 'https://en.wikipedia.org/wiki/2020_United_States_presidential_election_in_'+state+'#Results_by_county'
#     df_all = pd.read_html(url)
#     for dfs in df_all:
#         try:
#             if all(target_frame.isin(dfs.keys())):
#                 df = dfs
#                 flag = 0
#             if all(target_frame1.isin(dfs.keys())):
#                 df = dfs
#                 flag = 1
#         except:
#             continue
    
#     first_words = [element.split()[0] for element in df['County', 'County']]

#     df['County', 'County'] = first_words

#     df = df.loc[df['County', 'County'] == county]
#     if flag == 0:
#         data = float(df['Jo Jorgensen Libertarian', '%'].to_numpy()[0][:-1])
#     elif flag == 1:
#         data = float(df['Jo Jorgensen Independent', '%'].to_numpy()[0][:-1])

#     return data / 100.0

def obtain_voting(county_fips, state_dict, state_county_dict):
    state = state_dict[county_fips[:2]]
    if len(state.split()) >= 2:
        state = state.split()
        if len(state) == 3:
            state = state[1] + '_' + state[2] + '_' + state[3]
        elif len(state) == 2:
            state = state[1] + '_' + state[2]
    county_dict = state_county_dict[county_fips[:2]]
    county = county_dict[county_fips]

    df = pd.read_csv('./data/voting_data/vote_lib_data.csv')

    # if county_fips[0] == '0':
    #     county_fips = county_fips[1:]
    
    data = df.loc[df['county fips'].astype(int) == int(county_fips)]['vote for libertarian rate'].to_numpy()
    # print(data.item())
    return data.item()



def obtain_behavior_data(county_fips, census_api_key):
    df = {}

    df['people living alone'] = [
        obtain_people_living_along(county_fips, census_api_key)
    ]
    df['elder living alone'] = [
        obtain_elder_living_along(county_fips, census_api_key)
    ]
    df['family with grandchild'] = [
        obtain_family_with_grandchild(county_fips, census_api_key)
    ]
    df['marry divorce ratio'] = [
        obtain_marry_divorce_ratio(county_fips, census_api_key)
    ]
    df['non-religious ratio'] = [obtain_religious(county_fips)]
    df['self-employed rate'] = [obtain_self_employ(county_fips)]

    df = pd.DataFrame(data=df)

    df.to_csv(f"./data/{county_fips}/agents_behavior_data.csv", index=False)


def obtain_behavior_data_single_file(county_fips_list, census_api_key,
                                     state_dict, county_dict, state_county_dict):
    state_name = []
    county_name = []
    people_alone = []
    elder_alone = []
    grandchild_family = []
    marry_divorce = []
    non_religous = []
    self_employ = []
    voting = []

    for county_fips in tqdm(county_fips_list):
        state = state_dict[county_fips[:2]]
        county = county_dict[county_fips]

        state_name.append(state)
        county_name.append(county)
        try:
            people_alone.append(
                obtain_people_living_along(county_fips, census_api_key))
        except:
            people_alone.append(np.nan)

        try:
            elder_alone.append(
                obtain_elder_living_along(county_fips, census_api_key))
        except:
            elder_alone.append(np.nan)

        try:
            grandchild_family.append(
                obtain_family_with_grandchild(county_fips, census_api_key))
        except:
            grandchild_family.append(np.nan)

        try:
            marry_divorce.append(
                obtain_marry_divorce_ratio(county_fips, census_api_key))
        except:
            marry_divorce.append(np.nan)

        try:
            non_religous.append(obtain_religious(county_fips))
        except:
            non_religous.append(np.nan)

        try:
            self_employ.append(obtain_self_employ(county_fips))
        except:
            self_employ.append(np.nan)

        try:
            voting.append(obtain_voting(county_fips, state_dict, state_county_dict))
        except:
            voting.append(np.nan)

    df = {}

    df['county fips'] = county_fips_list
    df['state'] = state_name
    df['county'] = county_name
    df['people living alone'] = people_alone
    df['elder living alone'] = elder_alone
    df['family with grandchild'] = grandchild_family
    df['marry divorce ratio'] = marry_divorce
    df['non-religious ratio'] = non_religous
    df['self-employed rate'] = self_employ
    df['voting for libertarian'] = voting

    df = pd.DataFrame(data=df)

    df.to_csv(f"./data/county_behavior_data_lib.csv", index=False)


if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)

    import data_utils
    county_fips_list = data_utils.counties['MN'] + data_utils.counties['GA']

    # Define your Census API key
    CENSUS_API_KEY = "7a25a7624075d46f112113d33106b6648f42686a"

    ################################################################
    ############### THIS IS FOR ABM DEMOGRAPHIC DATA ###############
    ################################################################

    # Define the county FIPS code
    # 27109 is Olmsted County, MN
    # 27049 is Goodhue County, MN
    # 27051 is Grant County, MN
    # 27053 is Hennepin County, MN


    # List of Michigan county codes
    michigan_county_codes = [
        "001", "003", "005", "007", "009", "011", "013", "015", "017", "019",
        "021", "023", "025", "027", "029", "031", "033", "035", "037", "039",
        "041", "043", "045", "047", "049", "051", "053", "055", "057", "059",
        "061", "063", "065", "067", "069", "071", "073", "075", "077", "079",
        "081", "083", "085", "087", "089", "091", "093", "095", "097", "099",
        "101", "103", "105", "107", "109", "111", "113", "115", "117", "119",
        "121", "123", "125", "127", "129", "131", "133", "135", "137", "139",
        "141", "143", "145", "147", "149", "151", "153", "155", "157", "159",
        "161", "163", "165"
    ]

    # Create a list of Michigan FIPS codes
    michigan_fips_codes = ["26" + code for code in michigan_county_codes]

    minnesota_fips_codes = [
        "27109", "27053", "27003", "27049", "27061", "27029", "27171",
        "27091", "27139", "27049", "27051", "27053", '27081', '27041', 
        '27155', '27167', '27019', '27065', '27031', '27045', '27133', '27089'
    ]

    for county_fips in michigan_fips_codes:
        # create directory if it does not exist
        if not os.path.exists(f"./data/{county_fips}"):
            os.makedirs(f"./data/{county_fips}")
        print(f'Creating data for {county_fips}')
        obtain_household_size_distribution(county_fips, CENSUS_API_KEY)
        obtain_age_distribution(county_fips, CENSUS_API_KEY)
        obtain_occupation_distribution(county_fips, CENSUS_API_KEY)
        print()

    # no need the rest of the code for now
    quit()

    ################################################################
    ############### THIS IS FOR COLLECTIVISM INDEX #################
    ################################################################

    url = "https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt"
    if os.path.exists('./data/fips.txt'):
        os.remove('./data/fips.txt')
    wget.download(url, out='./data')

    with open('./data/fips.txt') as f:
        state_file = f.readlines()[16:67]
    with open('./data/fips.txt') as f:
        county_file = f.readlines()[72:3267]

    state_dict = {}
    for txt in state_file:
        data = txt.split()
        if len(data) == 2:
            state_dict[data[0]] = data[1].lower().capitalize()
        elif len(data) == 3:
            state_dict[data[0]] = data[1].lower().capitalize() + ' ' + data[2].lower().capitalize()
        elif len(data) == 4:
            state_dict[data[0]] = data[1].lower().capitalize() + ' ' + data[2].lower().capitalize() + ' ' + data[3].lower().capitalize()
        else:
            raise Exception('state name length invalid: {}'.format(data))

    county_dict = {}
    for txt in county_file:
        data = txt.split()
        county_dict[data[0]] = data[1]

    state_county_dict = {}
    for state_key in state_dict.keys():
        temp_dict = {}
        for key, value in county_dict.items():
            if int(key) > int(state_key)*1000 and int(key) < (int(state_key)+1)*1000:
                temp_dict[key] = value
        state_county_dict[state_key] = temp_dict

    county_fips_list = []

    for state in state_county_dict.keys():
        county_fips_list = county_fips_list + list(state_county_dict[state].keys())

    # print(county_fips_list)

    # county_fips_list = list(state_county_dict['02'].keys())

    # raise Exception('stop')

    url = "https://www.usreligioncensus.org/sites/default/files/2023-06/2020_USRC_Summaries.xlsx"
    if os.path.exists('./data/2020_USRC_Summaries.xlsx'):
        os.remove('./data/2020_USRC_Summaries.xlsx')
    wget.download(url, out='./data')
    obtain_behavior_data_single_file(county_fips_list, CENSUS_API_KEY,
                                     state_dict, county_dict, state_county_dict)

    if os.path.exists('./data/fips.txt'):
        os.remove('./data/fips.txt')

    if os.path.exists('./data/2020_USRC_Summaries.xlsx'):
        os.remove('./data/2020_USRC_Summaries.xlsx')