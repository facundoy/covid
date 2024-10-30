import os
import pandas as pd

# Define the path to the "data" folder
folder_path = "data/27003"

agents_ages_csv = os.path.join(folder_path, "agents_ages.csv")
agents_household_sizes_csv = os.path.join(folder_path, "agents_household_sizes.csv")
agents_occupations_csv = os.path.join(folder_path, "agents_occupations.csv")

df_ages = pd.read_csv(agents_ages_csv)
df_house = pd.read_csv(agents_household_sizes_csv)
df_occ = pd.read_csv(agents_occupations_csv)

count = 0
for index, row in df_ages.iterrows():
    # Access specific columns by name
    age = row['Age']
    number = row['Number']
    count += number
    # print(f"Age: {age}, Number: {number}")
print(f"Total in Ages = {count}")

count = 0
for index, row in df_house.iterrows():
    # Access specific columns by name
    houseSize = row['Household Size']
    number = row['Number']
    count += houseSize * number
print(f"Total in Household = {count}")

count = 0
for index, row in df_occ.iterrows():
    # Access specific columns by name
    occupation = row['Occupation']
    number = row['Number']
    count += number
print(f"Total in Occupation = {count}")