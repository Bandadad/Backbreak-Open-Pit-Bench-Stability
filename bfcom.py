import pandas as pd
import matplotlib.pyplot as plt

# Variables
PS_POC = 0.75  # Plane Shear Probability of Occurrence
WF_POC = 0.125 # Joint Probability of Occurrence of Wedge Failures

# File paths
bplane_file = 'BPlane_out.csv'
bwedge_file = 'BWedge_out.csv'

# Read in CSV files
bplane_df = pd.read_csv(bplane_file)
bwedge_df = pd.read_csv(bwedge_file)

# Ensure both DataFrames have the same number of cells
max_cells = max(len(bplane_df), len(bwedge_df))

# Extend the shorter DataFrame and fill "Probability of Stability" with 1.0 but keep original "Distance from Crest"
if len(bplane_df) < max_cells:
    bplane_df = bplane_df.reindex(range(max_cells))
    bplane_df['Probability of Stability'] = bplane_df['Probability of Stability'].fillna(1.0)

if len(bwedge_df) < max_cells:
    bwedge_df = bwedge_df.reindex(range(max_cells))
    bwedge_df['Probability of Stability'] = bwedge_df['Probability of Stability'].fillna(1.0)

# Compute probabilities of stability for each cell
PS = bplane_df['Probability of Stability']
PW = bwedge_df['Probability of Stability']

# Compute joint probability of failure for each cell
joint_pof = (1 - (PS_POC * WF_POC / (PS_POC + WF_POC))) * (PS_POC * (1 - PS) + WF_POC * (1 - PW))

# Convert back to Probability of Stability
joint_pos = 1 - joint_pof

# Ensure "Distance from Crest" is properly filled with values from the original dataset
distance_from_crest = bplane_df['Distance from Crest'].combine_first(bwedge_df['Distance from Crest'])

# Create a DataFrame for plotting
df_plot = pd.DataFrame({
    'Distance from Crest (m)': distance_from_crest,
    'Probability of Stability': joint_pos
})
print(df_plot)
# Plot Probability of Stability vs Distance from Crest
plt.figure(figsize=(8, 6))
plt.plot(df_plot['Distance from Crest (m)'], df_plot['Probability of Stability'], marker='o')
plt.title('Probability of Stability vs Distance from Crest')
plt.xlabel('Distance from Crest (m)')
plt.xlim(0, 25)
plt.ylim(0, 1.0)
plt.ylabel('Probability of Stability')
plt.grid(True)
plt.show()

