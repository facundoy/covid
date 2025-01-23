import pandas as pd
import glob
import os

# Step 1: Load the static household network
household_network = pd.read_csv("generated_networks/HOUSEHOLD_NETWORK.csv", header=None, names=["node1", "node2"])
household_network["normalized_pair"] = household_network.apply(lambda row: tuple(sorted((row["node1"], row["node2"]))), axis=1)

# Step 2: Initialize paths
base_path = "generated_networks"
occnet_path = os.path.join(base_path, "occnets")
schoolnet_path = os.path.join(base_path, "schoolnets")
output_path = "generated_networks/combined_networks"

# Step 3: Process each time step
for t in range(10):  # Assuming time steps 0 to 9
    combined_edges = set(household_network["normalized_pair"])  # Start with household network edges

    # Load occupation networks for the current time step
    occ_files = glob.glob(os.path.join(occnet_path, f"*step_{t}.csv"))
    for occ_file in occ_files:
        occ_df = pd.read_csv(occ_file, header=None, names=["node1", "node2"])
        occ_df["normalized_pair"] = occ_df.apply(lambda row: tuple(sorted((row["node1"], row["node2"]))), axis=1)
        combined_edges.update(occ_df["normalized_pair"])
    
    # Load the school network for the current time step
    school_file = os.path.join(schoolnet_path, f"School_step_{t}.csv")
    school_df = pd.read_csv(school_file, header=None, names=["node1", "node2"])
    school_df["normalized_pair"] = school_df.apply(lambda row: tuple(sorted((row["node1"], row["node2"]))), axis=1)
    combined_edges.update(school_df["normalized_pair"])

    # Convert combined edges to DataFrame and save
    combined_df = pd.DataFrame(list(combined_edges), columns=["node1", "node2"])
    combined_df.to_csv(os.path.join(output_path, f"Combined_step_{t}.csv"), index=False)
