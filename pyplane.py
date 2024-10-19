import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


def simulate_fracture(sim, N, dx, correlation_length, mean_spacing, std_spacing, 
                      mean_dip, std_dip, mean_ddr, std_ddr, 
                      bench_height, cell_width, mean_length, rock_density, 
                      mean_friction_angle, std_friction_angle, 
                      mean_cohesion, std_cohesion, dip_dir_VP):
    
    # Seed the random number generator for consistency (optional)
    np.random.seed(42 + sim)  # Different seed for each simulation

    # Generate correlated fields for spacing, dip, and dip direction
    field_spacing = generate_correlated_field(N, dx, correlation_length, mean_spacing, std_spacing)
    field_dip = generate_correlated_field(N, dx, correlation_length, mean_dip, std_dip)
    field_dip_dir = generate_correlated_field(N, dx, correlation_length, mean_ddr, std_ddr)

    # Ensure spacing is positive without altering the mean
    field_spacing[field_spacing <= 0] = 0.01  # Set negative spacings to a small positive value

    # Clip dips to realistic values (5 to 85 degrees to avoid extreme angles)
    field_dip = np.clip(field_dip, 5, 85)

    # Starting distance calculation
    uo = np.random.uniform(0, 1)  # Uniform random number U[0,1]
    SMU = mean_spacing
    FID = np.zeros((3, N))
    FID[0, 0] = uo * SMU  # Starting distance from bench toe

    # Mean dip DMU
    DMU = mean_dip

    # Convert angles to radians for trigonometric functions
    A_rad = np.radians(90)  # Dip of the bench face, vertical plane (VP)
    DMU_rad = np.radians(DMU)
    cos_A_minus_DMU = np.cos(A_rad - DMU_rad)
    if np.abs(cos_A_minus_DMU) < 1e-6:
        cos_A_minus_DMU = 1e-6  # Avoid division by zero

    sin_A = np.sin(A_rad)

    # Initialize arrays to store values for fractures
    probabilities = []
    x_crest = []
    FID_filtered = []
    prob_sliding = []  # Probability of sliding for each fracture
    prob_L_exceeds_required = []  # Probability that L > required length
    prob_stability = []  # Probability of stability for each fracture

    # Loop through fractures
    k = 0
    while True:
        if k == 0:
            FID[0, k] = uo * SMU  # Starting distance for the first fracture
        else:
            SPA_j = field_spacing[k - 1]  # Spacing between fractures
            FID[0, k] = FID[0, k - 1] + SPA_j / cos_A_minus_DMU

        if FID[0, k] >= bench_height or k >= N:
            break  # Stop if the fracture position exceeds the bench height or array limit

        # FID[1,k] - Dip of the fracture
        FID[1, k] = field_dip[k]

        # Calculate FID[2,k] - Required length for the fracture to reach the bench top
        dip_rad = np.radians(FID[1, k])
        sin_dip = np.sin(dip_rad)
        if np.abs(sin_dip) < 1e-6:
            sin_dip = 1e-6  # Avoid division by zero

        FID[2, k] = (bench_height - FID[0, k]) / sin_dip  # Required length to reach bench crest

        # Calculate x-coordinate at bench crest with probabilistic dip direction
        angle_diff = np.radians(abs(dip_dir_VP - field_dip_dir[k]))  # Difference from Dip direction of VP 
        x_crest_k = FID[2, k] * np.cos(dip_rad) * np.cos(angle_diff)  # Correct cosine calculation

        # Calculate probability of sufficient length
        P_L = np.exp(-FID[2, k] / mean_length)
        probabilities.append(P_L)
        x_crest.append(x_crest_k)
        FID_filtered.append([FID[0, k], FID[1, k], FID[2, k]])
        prob_L_exceeds_required.append(P_L)

        # Calculate probability of sliding using limit equilibrium method (LEM)
        phi_points = [mean_friction_angle - np.sqrt(3) * std_friction_angle, mean_friction_angle + np.sqrt(3) * std_friction_angle]
        c_points = [mean_cohesion - np.sqrt(3) * std_cohesion, mean_cohesion + np.sqrt(3) * std_cohesion]
        weights = [0.25, 0.25, 0.25, 0.25]  # Corresponding weights for FS values

        area = FID[2, k] * 1  # ft², assuming unit width

        height_of_block = bench_height - FID[0, k]
        base_of_block = x_crest_k 
        block_volume = 0.5 * base_of_block * height_of_block * 1 # Assumes unit width

        W = block_volume * rock_density  # Weight of the block in pounds

        FS_values = []
        for phi in phi_points:
            for c in c_points:
                FS = (c * area + W * np.cos(dip_rad) * np.tan(np.radians(phi))) / (W * np.sin(dip_rad))
                FS_values.append(FS)

        FS_mean = np.mean(FS_values)
        FS_std = np.std(FS_values)

        # Calculate probability of sliding (FS <= 1)
        if FS_std == 0:
            Pf = 1.0 if FS_mean <= 1 else 0.0
        else:
            Pf = norm.cdf((1 - FS_mean) / FS_std)
        prob_sliding.append(Pf)

        # Calculate probability of stability
        P_stability = 1 - (P_L * Pf)
        prob_stability.append(P_stability)

        k += 1  # Move to next fracture

    # Convert results to arrays and return
    probabilities = np.array(probabilities)
    x_crest = np.array(x_crest)
    prob_sliding = np.array(prob_sliding)
    prob_L_exceeds_required = np.array(prob_L_exceeds_required)
    prob_stability = np.array(prob_stability)
    FID_filtered = np.array(FID_filtered).T  # Transpose to match original shape

    # Return fracture data
    return {
        'x_crest': x_crest,
        'probabilities': probabilities,
        'prob_sliding': prob_sliding,
        'prob_L_exceeds_required': prob_L_exceeds_required,
        'prob_stability': prob_stability,
        'FID_filtered': FID_filtered,
    }


def process_simulations(Ns, N, dx, correlation_length, mean_spacing, std_spacing, 
                        mean_dip, std_dip, mean_ddr, std_ddr, 
                        bench_height, cell_width, mean_length, rock_density, 
                        mean_friction_angle, std_friction_angle, 
                        mean_cohesion, std_cohesion, dip_dir_VP):
    """
    Runs multiple simulations, aggregates the results, and returns a master dataframe.

    Args:
        Ns (int): Number of simulations.
        N (int): Number of points (realizations) in each simulation.
        dx (float): Spatial step size.
        correlation_length (float): Correlation length for the random field.
        mean_spacing (float): Mean fracture spacing.
        std_spacing (float): Standard deviation of fracture spacing.
        mean_dip (float): Mean dip of fractures.
        std_dip (float): Standard deviation of fracture dip.
        mean_ddr (float): Mean dip direction of fractures.
        std_ddr (float): Standard deviation of fracture dip direction.
        bench_height (float): Height of the bench face.
        cell_width (float): Width of each backbreak cell.
        mean_length (float): Mean fracture length.
        rock_density (float): Density of the rock (lb/ft³).
        mean_friction_angle (float): Mean friction angle (degrees).
        std_friction_angle (float): Standard deviation of friction angle (degrees).
        mean_cohesion (float): Mean cohesion (psf).
        std_cohesion (float): Standard deviation of cohesion (psf).

    Returns:
        pd.DataFrame: A master dataframe containing the aggregated results of all simulations.
    """
    
    # Initialize an empty dataframe to store the results of all simulations
    master_df = pd.DataFrame()

    # Loop through each simulation
    for sim in range(Ns):
        print(f"Running simulation {sim + 1}/{Ns}")

        # Call simulate_fracture for each simulation
        result = simulate_fracture(
            sim, N, dx, correlation_length, mean_spacing, std_spacing, 
            mean_dip, std_dip, mean_ddr, std_ddr, bench_height, 
            cell_width, mean_length, rock_density,  
            mean_friction_angle, std_friction_angle, mean_cohesion, std_cohesion, dip_dir_VP
        )
        
        # Extract fracture data from the result
        x_crest = result['x_crest']
        prob_stability = result['prob_stability']
        prob_L_exceeds_required = result['prob_L_exceeds_required']
        prob_sliding = result['prob_sliding']
        FID_filtered = result['FID_filtered']

        # Number of fractures in this simulation
        num_fractures = len(x_crest)
        
        # Create a DataFrame with the calculated values for the current simulation
        data = {
            'Distance from Crest': x_crest,  # Horizontal positions at bench crest for each fracture
            'Probability of Stability': prob_stability,
            'Simulation': sim + 1,  # Simulation number
            'Probability of Length Exceeds Required': prob_L_exceeds_required,
            'Probability of Sliding': prob_sliding
        }
        
        df_sim = pd.DataFrame(data)

        # Assign Cell Number based on Distance from Crest
        df_sim['Cell Number'] = (df_sim['Distance from Crest'] // cell_width).astype(int) + 1

        # Append the result of this simulation to the master dataframe
        master_df = pd.concat([master_df, df_sim], ignore_index=True)

    # Return the master dataframe containing results of all simulations
    return master_df


def fill_missing_cells(df_grouped, cell_number, cell_width):
    # Generate a complete sequence of Cell Numbers
    all_cells = pd.DataFrame({'Cell Number': range(1, cell_number + 1)})
    
    # Merge with the existing data to find missing Cell Numbers
    df_filled = pd.merge(all_cells, df_grouped, on='Cell Number', how='left')
    
    # For rows with missing values, fill with specified values
    df_filled['Probability of Stability'] = df_filled['Probability of Stability'].fillna(1.0)
    df_filled['Distance from Crest'] = df_filled['Cell Number'] * cell_width - 0.5 * cell_width
    
    return df_filled


# Function to generate a correlated random field using FFT
def generate_correlated_field(N, dx, lc, mean, std):
    dk = 1 / (N * dx)  # Frequency step size
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Angular frequencies

    # Power Spectral Density (PSD) for exponential covariance function
    S_k = 2 * lc / (1 + (lc * k) ** 2)

    # Generate random phases
    random_phases = np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N)

    # Fourier coefficients
    fourier_coefficients = np.sqrt(S_k) * random_phases

    # Inverse FFT to get the correlated field
    field = np.fft.ifft(fourier_coefficients).real

    # Standardize and scale to desired mean and std
    field = (field - np.mean(field)) / np.std(field)
    field = field * std + mean

    return field


# Define the function to calculate cell stability
def calculate_cell_stability(group, master_df):
    # N is the number of simulations where the Cell Number exists in the group (at least one failure path)
    N = group['Simulation'].nunique()  # Number of unique simulations with this Cell Number
    
    if N == 0:
        # If no simulations contain this Cell Number, set Probability of Stability to 1.0
        return pd.Series({'Probability of Stability': 1.0})
    
    # Total number of bench simulations (Ns or N_T)
    Nt = len(master_df['Simulation'].unique())
    
    # Calculate the summation of products for (1 - P_Lj) for each simulation Si
    stability_product_sum = 0
    for si_value in group['Simulation'].unique():
        group_si = group[group['Simulation'] == si_value]
        
        # Calculate the product (1 - P_Lj) for all failure paths (J_i) in this Si
        product_term = 1
        for index, row in group_si.iterrows():
            P_Lj = row['Probability of Length Exceeds Required']  # This is the probability of sufficient length (P_Lj)
            P_Sj = row['Probability of Sliding']  # This is the probability of sliding (P_Sj)
            product_term *= (1 - P_Lj) + P_Lj * (1 - P_Sj)  # Apply the product for each failure path
        
        # Add the product for this simulation to the total summation
        stability_product_sum += product_term
    
    # Apply the formula for Probability of Stability
    prob_stability = ((Nt - N) / Nt) + (1 / Nt) * stability_product_sum
    
    return pd.Series({'Probability of Stability': prob_stability})


def plot_probability_of_stability(df_grouped, width):
    """
    Plots Probability of Stability vs Distance from Crest.

    Args:
        df_grouped (pd.DataFrame): DataFrame containing the grouped simulation results.
        width (float): Width of the bench.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(df_grouped['Distance from Crest'], df_grouped['Probability of Stability'], marker='o')
    plt.title('Probability of Stability vs Distance from Crest (Aggregated Simulations)')
    plt.xlabel('Distance from Crest (ft)')
    plt.xlim(0, width)
    plt.ylim(0, 1.0)
    plt.ylabel('Probability of Stability')
    plt.grid(True)
    plt.show()


def plot_fractures(FID_filtered, num_fractures, bench_height, width):
    """
    Plots the fractures within the bench height and width.

    Args:
        FID_filtered (np.array): Array of filtered fracture data (position, dip, length).
        num_fractures (int): Number of fractures to plot.
        bench_height (float): Height of the bench.
        width (float): Width of the bench.
    """
    plt.figure(figsize=(8, bench_height / width * 8))
    
    # Plot each fracture line
    for i in range(num_fractures):
        y_start = FID_filtered[0, i]  # Starting vertical position
        dip = FID_filtered[1, i]  # Fracture dip
        length = FID_filtered[2, i]  # Fracture length

        # Calculate x and y coordinates of the fracture line
        x_start = 0  # Starting at the bench face
        dip_rad = np.radians(dip)
        x_end = x_start + length * np.cos(dip_rad)
        y_end = y_start + length * np.sin(dip_rad)

        plt.plot([x_start, x_end], [y_start, y_end], 'r-')

    # Draw the bench face
    plt.plot([0, 0], [0, bench_height], 'k-', linewidth=2)

    plt.xlabel('Bench Width (ft)')
    plt.ylabel('Bench Height (ft)')
    plt.title('Simulated Fractures within Bench Width')
    plt.xlim(0, width)
    plt.ylim(0, bench_height)
    plt.grid(True)
    plt.show()


def plot_probability_of_exceeding_length(x_crest, prob_L_exceeds_required):
    """
    Plots Probability that Fracture Length exceeds the Required Length vs Horizontal Position.

    Args:
        x_crest (np.array): Horizontal positions along the bench crest.
        prob_L_exceeds_required (np.array): Probabilities that the fracture length exceeds the required length.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x_crest, prob_L_exceeds_required, 'bo-')
    plt.xlabel('Horizontal Position along Bench Crest (ft)')
    plt.ylabel('Probability P[L > Required Length]')
    plt.title('Probability of Fracture Length Exceeding Required Length')
    plt.grid(True)
    plt.show()


def plot_probability_of_sliding(x_crest, prob_sliding):
    """
    Plots Probability of Sliding vs Horizontal Position along Bench Crest.

    Args:
        x_crest (np.array): Horizontal positions along the bench crest.
        prob_sliding (np.array): Probabilities of sliding for each fracture.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x_crest, prob_sliding, 'ro-')
    plt.xlabel('Horizontal Position along Bench Crest (ft)')
    plt.ylabel('Probability of Sliding')
    plt.title('Probability of Sliding vs Position along Bench Crest')
    plt.grid(True)
    plt.show()


def main():
    # User input and main logic
    # Define simulation parameters
    Ns = 20 # Number of bench simulations to run
    N = 256 # Mumber of fracture realizations to attempt per simulation

    
    # Define the bench face orientation and dimensions  - vertical plane (VP)
    dip_VP = 90
    dip_dir_VP = 195
    height = 25 # ft
    width = height # ft 
    cell_number = 12
    cell_width = width / cell_number
    rock_density = 165 # pcf
 
    mean_dip, std_dip = 55, 8
    mean_ddr, std_ddr = 180, 5
    mean_spacing = 2.5 # ft
    std_spacing = 0.5
    mean_friction_angle = 35
    std_friction_angle = 2
    mean_cohesion = 0.00 # psf
    std_cohesion = 0.0
    mean_length = 11 # ft

    correlation_length = 10

     # Define dx (spatial step size)
    dx = height / N  # Spatial step size

    master_df = process_simulations(Ns, N, dx, correlation_length, mean_spacing, std_spacing, 
                                    mean_dip, std_dip, mean_ddr, std_ddr, 
                                    height, cell_width, mean_length, rock_density, 
                                    mean_friction_angle, std_friction_angle, 
                                    mean_cohesion, std_cohesion, dip_dir_VP)

    # Group by Cell Number and calculate stability
    df_grouped = master_df.groupby('Cell Number').apply(lambda group: calculate_cell_stability(group, master_df), include_groups=False).reset_index()
    df_grouped['Distance from Crest'] = (df_grouped['Cell Number'] * cell_width) - 0.5 * cell_width
    df_grouped = fill_missing_cells(df_grouped, cell_number, cell_width)
    df_grouped.to_csv('pyplane_out.csv', index=False)
    
    # Plot results
    plot_probability_of_stability(df_grouped, width)
    # Other plots can be added similarly...see function list
    
if __name__ == "__main__":
    main()


# Optional: Print probabilities and positions
# for i in range(num_fractures):
#     print(f"Fracture {i+1}: x_crest = {x_crest[i]:.2f} m, "
#           f"Required Length = {FID_filtered[2, i]:.2f} m, "
#           f"P[L > Required Length] = {prob_L_exceeds_required[i]:.4f}, "
#           f"Probability of Sliding = {prob_sliding[i]:.4f}, "
#           f"Probability of Stability = {prob_stability[i]:.4f}")
