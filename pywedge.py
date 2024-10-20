import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wedge_functions import process_wedges


# Function to calculate wedge parameters and factor of safety
def calculate_wedge_params(row, c1_mean, c1_std, phi1_mean, phi1_std, c2_mean, c2_std, phi2_mean, phi2_std, dip_dir_VP, gamma):
    # Extract values from the row
    dip1 = row['dip_JP1']
    dip2 = row['dip_JP2']
    alpha1 = row['dip_dir_JP1']
    alpha2 = row['dip_dir_JP2']
    H1 = row['Wedge Height']
    
    # Call the updated process_wedges function with the passed parameters
    trend_i, plunge_i, distance_from_crest, FOS, probability_of_failure, message = process_wedges(
        dip1, alpha1, c1_mean, c1_std, phi1_mean, phi1_std, dip2, alpha2, c2_mean, c2_std, phi2_mean, phi2_std, H1, dip_dir_VP, gamma
    )

    # Return the calculated values along with the message
    return pd.Series([trend_i, plunge_i, distance_from_crest, FOS, probability_of_failure, message])


# Function to compute the normal vector from dip and dip direction
def compute_normal(dip, dip_dir):
    dip_rad = np.radians(dip)
    dip_dir_rad = np.radians(dip_dir)
    n_x = np.sin(dip_rad) * np.sin(dip_dir_rad)
    n_y = np.sin(dip_rad) * np.cos(dip_dir_rad)
    n_z = np.cos(dip_rad)
    return np.array([n_x, n_y, n_z])


# Function to compute the line of intersection between two planes
def plane_intersection(n1, d1, n2, d2):
    l = np.cross(n1, n2)
    if np.linalg.norm(l) < 1e-6:
        return None, None  # Planes are parallel or identical
    A = np.array([n1, n2, l]).T
    b = -np.array([d1, d2, 0])
    try:
        point = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None
    return point, l


# Function to compute intersection of two lines in the plane
def compute_line_intersection(s1, t1, ds1, dt1, s2, t2, ds2, dt2):
    A = np.array([[ds1, -ds2], [dt1, -dt2]])
    b = np.array([s2 - s1, t2 - t1])
    try:
        taus = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None  # Lines are parallel, no intersection
    tau1, tau2 = taus
    intersection_x = s1 + tau1 * ds1
    intersection_y = t1 + tau1 * dt1
    return (intersection_x, intersection_y)


# Function to generate planes at specified spacing
def generate_planes(normals, spacing, count):
    planes = []
    num_planes = len(normals)
    for i in range(num_planes):
        n = normals[i]
        n = n / np.linalg.norm(n)  # Ensure normal vector is unit length
        d = - (i - count) * spacing
        planes.append((n, d))
    return planes


def generate_joint_planes(dip_mean, dip_std, dip_dir_mean, dip_dir_std, spacing, width):
    # Calculate the number of planes
    count = int(width / spacing)
    num_planes = 2 * count + 1

    # Generate random dips and dip directions
    dips = np.random.normal(dip_mean, dip_std, size=num_planes)
    dip_dirs = np.random.normal(dip_dir_mean, dip_dir_std, size=num_planes)
    
    # Compute the normal vectors for the planes
    normals = [compute_normal(dip, dip_dir) for dip, dip_dir in zip(dips, dip_dirs)]

    # Generate the planes using the normal vectors and spacing
    planes = generate_planes(normals, spacing, count)

    return planes, dips, dip_dirs


# Function to check if line intersects the top edge
def line_intersects_top_edge(s0, t0, ds, dt, tau_min, tau_max, width, height):
    if dt != 0:
        tau_top = (height/2 - t0) / dt
        if tau_min <= tau_top <= tau_max or tau_max <= tau_top <= tau_min:
            s_top = s0 + tau_top * ds
            if -width/2 <= s_top <= width/2:
                return True
    else:
        if t0 == height/2 and (-width/2 <= s0 <= width/2):
            return True
    return False


def calculate_tau_limits(s0, t0, ds, dt, width, height):
    # Calculate tau for s-axis
    tau_s1 = (-width/2 - s0) / ds if ds != 0 else -np.inf
    tau_s2 = (width/2 - s0) / ds if ds != 0 else np.inf

    # Calculate tau for t-axis
    tau_t1 = (-height/2 - t0) / dt if dt != 0 else -np.inf
    tau_t2 = (height/2 - t0) / dt if dt != 0 else np.inf

    # Determine minimum and maximum tau
    tau_min = max(min(tau_s1, tau_s2), min(tau_t1, tau_t2))
    tau_max = min(max(tau_s1, tau_s2), max(tau_t1, tau_t2))

    return tau_min, tau_max


def generate_intersections(planes_JP1, dips_JP1, dip_dirs_JP1, planes_JP2, dips_JP2, dip_dirs_JP2, dip_VP, dip_dir_VP, width, height):
    # Compute n_VP and d_VP within the function
    n_VP = compute_normal(dip_VP, dip_dir_VP)
    d_VP = 0  # Passing through origin
    
    # Calculate u1 and u2 based on the normal vector of the vertical plane (n_VP)
    u1 = np.cross(n_VP, np.array([0, 0, 1]))
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(u1, n_VP)
    u2 /= np.linalg.norm(u2)
    
    # Initialize lists to store data
    intersection_points = []
    filtered_lines_JP1 = []
    filtered_lines_JP2 = []

    # Process Joint Set 1
    for idx1, (n_JP1, d_JP1) in enumerate(planes_JP1):
        dip_JP1 = dips_JP1[idx1]
        dip_dir_JP1 = dip_dirs_JP1[idx1]
        point1, direction1 = plane_intersection(n_VP, d_VP, n_JP1, d_JP1)
        if point1 is None:
            continue
        s0_1 = np.dot(point1, u1)
        t0_1 = np.dot(point1, u2)
        ds1 = np.dot(direction1, u1)
        dt1 = np.dot(direction1, u2)
        if ds1 == 0 and dt1 == 0:
            continue
        tau_min, tau_max = calculate_tau_limits(s0_1, t0_1, ds1, dt1, width, height)
        if tau_min > tau_max or not line_intersects_top_edge(s0_1, t0_1, ds1, dt1, tau_min, tau_max, width, height):
            continue
        filtered_lines_JP1.append({'s0': s0_1, 't0': t0_1, 'ds': ds1, 'dt': dt1, 'dip': dip_JP1, 'dip_dir': dip_dir_JP1})

    # Process Joint Set 2 and compute intersections with Joint Set 1
    for idx2, (n_JP2, d_JP2) in enumerate(planes_JP2):
        dip_JP2 = dips_JP2[idx2]
        dip_dir_JP2 = dip_dirs_JP2[idx2]
        point2, direction2 = plane_intersection(n_VP, d_VP, n_JP2, d_JP2)
        if point2 is None:
            continue
        s0_2 = np.dot(point2, u1)
        t0_2 = np.dot(point2, u2)
        ds2 = np.dot(direction2, u1)
        dt2 = np.dot(direction2, u2)
        if ds2 == 0 and dt2 == 0:
            continue
        tau_min2, tau_max2 = calculate_tau_limits(s0_2, t0_2, ds2, dt2, width, height)
        if tau_min2 > tau_max2 or not line_intersects_top_edge(s0_2, t0_2, ds2, dt2, tau_min2, tau_max2, width, height):
            continue
        filtered_lines_JP2.append({'s0': s0_2, 't0': t0_2, 'ds': ds2, 'dt': dt2, 'dip': dip_JP2, 'dip_dir': dip_dir_JP2})

        for line_JP1 in filtered_lines_JP1:
            intersection = compute_line_intersection(line_JP1['s0'], line_JP1['t0'], line_JP1['ds'], line_JP1['dt'], s0_2, t0_2, ds2, dt2)
            if intersection is not None:
                x, y = intersection
                if -width/2 <= x <= width/2 and -height/2 <= y <= height/2:
                    intersection_points.append({
                        'X (ft)': x,
                        'Y (ft)': y,
                        'dip_JP1': line_JP1['dip'],
                        'dip_dir_JP1': line_JP1['dip_dir'],
                        'dip_JP2': dip_JP2,
                        'dip_dir_JP2': dip_dir_JP2
                    })

    if not intersection_points:
        print("No valid intersections found for this simulation.")
        return [], [], []  # Return empty lists to handle this case
    else:
        return filtered_lines_JP1, filtered_lines_JP2, intersection_points


def plot_joints_and_intersections(filtered_lines_JP1, filtered_lines_JP2, intersection_points, width, height, simulation_number):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-width/2, width/2)
    ax.set_ylim(-height/2, height/2)
    ax.set_xlabel('X-axis (ft)')
    ax.set_ylabel('Y-axis (ft)')
    ax.set_title(f'Projection of Joint Sets on Vertical Plane for Simulation {simulation_number} \n(Only lines intersecting the top edge are shown)')

    # Draw rectangle boundary
    rect_corners = np.array([[-width/2, -height/2], [width/2, -height/2], [width/2, height/2], [-width/2, height/2], [-width/2, -height/2]])
    ax.plot(rect_corners[:, 0], rect_corners[:, 1], 'k-')

    # Plot Joint Set 1
    for line in filtered_lines_JP1:
        tau_min, tau_max = calculate_tau_limits(line['s0'], line['t0'], line['ds'], line['dt'], width, height)
        s1 = line['s0'] + tau_min * line['ds']
        t1 = line['t0'] + tau_min * line['dt']
        s2 = line['s0'] + tau_max * line['ds']
        t2 = line['t0'] + tau_max * line['dt']
        ax.plot([s1, s2], [t1, t2], 'b-')

    # Plot Joint Set 2
    for line in filtered_lines_JP2:
        tau_min, tau_max = calculate_tau_limits(line['s0'], line['t0'], line['ds'], line['dt'], width, height)
        s1 = line['s0'] + tau_min * line['ds']
        t1 = line['t0'] + tau_min * line['dt']
        s2 = line['s0'] + tau_max * line['ds']
        t2 = line['t0'] + tau_max * line['dt']
        ax.plot([s1, s2], [t1, t2], 'r-')

    plt.grid(True)
    plt.show()


def plot_FOS_histogram(df_intersections):
    plt.figure(figsize=(8, 6))
    plt.hist(df_intersections['Factor of Safety'], bins=50, color='blue', edgecolor='black')
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='FOS = 1')
    plt.title('Histogram of Factor of Safety Values')
    plt.xlabel('Factor of Safety')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def plot_POS_histogram(df_intersections):
    plt.figure(figsize=(8, 6))
    plt.hist(df_intersections['Prob of Sliding'], bins=50, color='blue', edgecolor='black')
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='FOS = 1')
    plt.title('Histogram of Probability of Sliding Values')
    plt.xlabel('Probability of Sliding')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def fill_missing_cells(df_grouped, cell_number, cell_width):
    # Generate a complete sequence of Cell Numbers
    all_cells = pd.DataFrame({'Cell Number': range(1, cell_number + 1)})
    
    # Merge with the existing data to find missing Cell Numbers
    df_filled = pd.merge(all_cells, df_grouped, on='Cell Number', how='left')
    
    # For rows with missing values, fill with specified values
    df_filled['Probability of Stability'] = df_filled['Probability of Stability'].fillna(1.0)
    df_filled['Distance from Crest'] = df_filled['Cell Number'] * cell_width - 0.5 * cell_width
    
    return df_filled


def process_dataframe(intersection_points, height, cell_width, mean_length1, mean_length2, c1_mean, c1_std, phi1_mean, phi1_std, c2_mean, c2_std, phi2_mean, phi2_std, dip_dir_VP, gamma):
    # Create the DataFrame
    df = pd.DataFrame(intersection_points)
    
    # Calculate additional columns
    df["Wedge Height"] = (height / 2) - df["Y (ft)"]
    
    # Use lambda to pass additional parameters into calculate_wedge_params
    df[['Trend', 'Plunge', 'Distance from Crest', 'Factor of Safety', 'Prob of Sliding', 'Message']] = df.apply(
        lambda row: calculate_wedge_params(row, c1_mean, c1_std, phi1_mean, phi1_std, c2_mean, c2_std, phi2_mean, phi2_std, dip_dir_VP, gamma), axis=1
    )
    
    df["Intersection Length"] = df["Wedge Height"] / np.sin(np.radians(df["Plunge"]))
    
    # Calculate cell number and probability P_L(wedge)
    # Handle NaN or infinite values in 'Distance from Crest'
    df['Distance from Crest'] = df['Distance from Crest'].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Distance from Crest'])

    df['Cell Number'] = (df['Distance from Crest'] // cell_width).astype(int) + 1
    df["P_L(wedge)"] = np.exp(-df["Intersection Length"] / mean_length1) * np.exp(-df["Intersection Length"] / mean_length2)
    df['Prob of Failure'] = df['Prob of Sliding'] * df['P_L(wedge)']
    
    # Move 'Message' column to the last position
    df = df[[col for col in df.columns if col != 'Message'] + ['Message']]
    
    return df


def calculate_cell_stability(group, master_df):
    # N is the number of simulations where the Cell Number exists in the group (at least one failure path)
    N = group['Si'].nunique()  # Number of unique simulations with this Cell Number
    
    if N == 0:
        # If no simulations contain this Cell Number, set Probability of Stability to 1.0
        return pd.Series({'Probability of Stability': 1.0})
    
    # Total number of bench simulations (Ns or N_T)
    Nt = len(master_df['Si'].unique())
    
    # Calculate the summation of products for (1 - P_Lj) for each simulation Si
    stability_product_sum = 0
    for si_value in group['Si'].unique():
        group_si = group[group['Si'] == si_value]
        
        # Calculate the product (1 - P_Lj) for all failure paths (J_i) in this Si
        product_term = 1
        for index, row in group_si.iterrows():
            P_Lj = row['P_L(wedge)']  # This is the probability of sufficient length (P_Lj)
            P_Sj = row['Prob of Sliding']  # This is the probability of sliding (P_Sj)
            product_term *= (1 - P_Lj) + P_Lj * (1 - P_Sj)  # Apply the product for each failure path
        
        # Add the product for this simulation to the total summation
        stability_product_sum += product_term
    
    # Apply the formula for Probability of Stability
    prob_stability = ((Nt - N) / Nt) + (1 / Nt) * stability_product_sum
    
    return pd.Series({'Probability of Stability': prob_stability})


def main():
    # Define the number of simulations
    Ns = 20 # Number of simulations

    # Define the bench face orientation - vertical plane (VP)
    dip_VP = 90
    dip_dir_VP = 195

    # Define the dimensions of the simulation window on the bench face
    height = 50
    width = height
    gamma = 165  # Unit weight of rock (lb/ft^3)

    # Define Backbreak Cells
    cell_number = 12
    cell_width = width / cell_number

    # Probabilistic sampling parameters for Joint Sets
    # Joint Set 1
    dip_JP1_mean, dip_JP1_std = 72.35, 8.35
    dip_dir_JP1_mean, dip_dir_JP1_std = 108.71, 11.7
    spacing_JP1 = 3.8
    mean_length1 = 8.5
    phi1_mean, phi1_std = 31, 7
    c1_mean, c1_std = 0, 0

    # Joint Set 2
    dip_JP2_mean, dip_JP2_std = 65.0, 13.87
    dip_dir_JP2_mean, dip_dir_JP2_std = 214.8, 18.44
    spacing_JP2 = 2.2
    mean_length2 = 8.5
    phi2_mean, phi2_std = 25, 5
    c2_mean, c2_std = 0, 0

    # Specify the simulation number to plot generated wedges on the simulation windown (set to None if not needed)
    simulation_number = 20  # Replace with the desired simulation number
    
    # Variables to store the data for the specific simulation for plotting later
    saved_filtered_lines_JP1 = None
    saved_filtered_lines_JP2 = None
    saved_intersection_points = None

    # Initialize an empty dataframe to store the results of all simulations
    master_df = pd.DataFrame()

    for i in range(Ns):
        print(f"Running simulation {i + 1}/{Ns}")
        
        # Generate planes and intersections
        planes_JP1, dips_JP1, dip_dirs_JP1 = generate_joint_planes(dip_JP1_mean, dip_JP1_std, dip_dir_JP1_mean, dip_dir_JP1_std, spacing_JP1, width)
        planes_JP2, dips_JP2, dip_dirs_JP2 = generate_joint_planes(dip_JP2_mean, dip_JP2_std, dip_dir_JP2_mean, dip_dir_JP2_std, spacing_JP2, width)
        filtered_lines_JP1, filtered_lines_JP2, intersection_points = generate_intersections(
            planes_JP1, dips_JP1, dip_dirs_JP1, planes_JP2, dips_JP2, dip_dirs_JP2, dip_VP, dip_dir_VP, width, height)

        # Check if intersection_points is empty
        if not intersection_points:
            print(f"No valid intersections found for simulation {i + 1}. Skipping.")
            continue  # Skip to the next simulation

        # Process the intersection points into a dataframe
        df_intersections = process_dataframe(
            intersection_points, height, cell_width, mean_length1, mean_length2, c1_mean, c1_std, phi1_mean, phi1_std,
            c2_mean, c2_std, phi2_mean, phi2_std, dip_dir_VP, gamma
        )
        # Uncomment to see a plot of the wedges being generated
        #plot_joints_and_intersections(filtered_lines_JP1, filtered_lines_JP2, intersection_points, width, height)
        
        # Add a column for the current simulation (Si)
        df_intersections['Si'] = i + 1

        # Append the result of this simulation to the master dataframe
        master_df = pd.concat([master_df, df_intersections], ignore_index=True)

        # If the current simulation is the desired one for plotting, save the data
        if simulation_number and (i + 1) == simulation_number:
            saved_filtered_lines_JP1 = filtered_lines_JP1
            saved_filtered_lines_JP2 = filtered_lines_JP2
            saved_intersection_points = intersection_points

    # Group by Cell Number and calculate the required values for aggregated simulations
    # Apply the calculation to each grouped dataframe
    df_grouped = master_df.groupby('Cell Number', group_keys=False).apply(
        lambda group: calculate_cell_stability(group, master_df), include_groups=False).reset_index()

    # Calculate Distance from Crest
    df_grouped['Distance from Crest'] = (df_grouped['Cell Number'] * cell_width) - 0.5 * cell_width
    df_grouped = fill_missing_cells(df_grouped, cell_number, cell_width)
    df_grouped.to_csv('pywedge_out.csv', index=False)
    plot_FOS_histogram(master_df)

    # Plot Probability of Stability vs Distance from Crest
    plt.figure(figsize=(8, 6))
    plt.plot(df_grouped['Distance from Crest'], df_grouped['Probability of Stability'], marker='o')
    plt.title('Probability of Stability vs Distance from Crest (Aggregated Simulations)')
    plt.xlabel('Distance from Crest (ft)')
    plt.xlim(0, width)
    plt.ylim(0, 1.0)
    plt.ylabel('Probability of Stability')
    plt.grid(True)
    plt.show()
    
    # Plot the joints and intersections for the specified simulation
    if saved_filtered_lines_JP1 is not None and saved_filtered_lines_JP2 is not None and saved_intersection_points is not None:
        plot_joints_and_intersections(saved_filtered_lines_JP1, saved_filtered_lines_JP2, saved_intersection_points, width, height, simulation_number)


if __name__ == "__main__":
    main()
