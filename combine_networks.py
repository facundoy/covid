import os
import pandas as pd
import glob
import re
from tqdm import tqdm

def extract_date_from_filename(filename):
    """Extracts the date (yyyy-mm-dd) from occupation or school network filenames."""
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return match.group(0) if match else None

def combine_networks(output_dir):
    # Identify all county folders (starting with "26___")
    county_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("26")]

    # Wrap the county loop with tqdm for a progress bar
    for county in tqdm(county_dirs, desc="Processing Counties", unit="county"):
        specific_dir = os.path.join(output_dir, county, "specific_networks")
        combined_dir = os.path.join(output_dir, county, "combined_networks")
        os.makedirs(combined_dir, exist_ok=True)  # Ensure output directory exists

        # Load the static household network
        household_file = os.path.join(specific_dir, "HOUSEHOLD_NETWORK.csv")
        household_network = pd.read_csv(household_file, header=None, names=["node1", "node2"])
        household_network["normalized_pair"] = household_network.apply(lambda row: tuple(sorted((row["node1"], row["node2"]))), axis=1)

        # Define network paths
        occnet_path = os.path.join(specific_dir, "occnets")
        schoolnet_path = os.path.join(specific_dir, "schoolnets")
        randnet_path = os.path.join(specific_dir, "randnets")

        # Identify time steps based on occupation or school networks
        occ_files = sorted(glob.glob(os.path.join(occnet_path, "*.csv")))
        school_files = sorted(glob.glob(os.path.join(schoolnet_path, "*.csv")))

        # Extract ordered dates from file names
        date_list = sorted(set(filter(None, [extract_date_from_filename(f) for f in occ_files + school_files])))

        # Wrap the time step loop with tqdm for a progress bar
        for t, date in tqdm(enumerate(date_list), total=len(date_list), desc=f"Processing {county} Time Steps", unit="step", leave=False):
            combined_edges = set(household_network["normalized_pair"])  # Start with household network edges
            
            # Load occupation networks for the current time step
            occ_files = glob.glob(os.path.join(occnet_path, f"*date_{date}.csv"))
            
            # Wrap occupation network file loop with tqdm
            for occ_file in tqdm(occ_files, desc=f"Loading Occupation Networks for {county}, Date: {date}", unit="file", leave=False):
                occ_df = pd.read_csv(occ_file, header=None, names=["node1", "node2"])
                occ_df["normalized_pair"] = occ_df.apply(lambda row: tuple(sorted((row["node1"], row["node2"]))), axis=1)
                combined_edges.update(occ_df["normalized_pair"])

            # Load the school network for the current time step
            school_file = os.path.join(schoolnet_path, f"School_date_{date}.csv")
            if os.path.exists(school_file):
                school_df = pd.read_csv(school_file, header=None, names=["node1", "node2"])
                school_df["normalized_pair"] = school_df.apply(lambda row: tuple(sorted((row["node1"], row["node2"]))), axis=1)
                combined_edges.update(school_df["normalized_pair"])

            # Load the random network for the current time step
            rand_file = os.path.join(randnet_path, f"random_time_{t}.csv")
            if os.path.exists(rand_file):
                rand_df = pd.read_csv(rand_file, header=None, names=["node1", "node2"])
                rand_df["normalized_pair"] = rand_df.apply(lambda row: tuple(sorted((row["node1"], row["node2"]))), axis=1)
                combined_edges.update(rand_df["normalized_pair"])

            # Convert combined edges to DataFrame and save
            combined_df = pd.DataFrame(list(combined_edges), columns=["node1", "node2"])
            output_filename = os.path.join(combined_dir, f"Combined_date_{date}.csv")
            combined_df.to_csv(output_filename, index=False)
            print(f"Saved: {output_filename}")

base_dir = "generated_networks"
combine_networks(base_dir)