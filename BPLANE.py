import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def distance_from_crest(h, plunge, trend, alpha4):
    # Convert plunge and alpha4 to radians
    plunge_rad = np.radians(plunge)
    trend_rad = np.radians(trend)
    alpha4_rad = np.radians(alpha4)
    
    cot_plunge = 1 / np.tan(plunge_rad)
    angle_diff = trend_rad - alpha4_rad
    
    distance = h * cot_plunge * np.cos(angle_diff)
    
    return distance

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

# User inputs

# Define the bench face orientation - vertical plane (VP)
dip_VP = 90
dip_dir_VP = 195

POC = 1.0 # probability of occurrence

bench_height = 20
bench_width = 100

mean_dip, std_dip = 55, 2
mean_ddr, std_ddr = 180, 5

mean_spacing = 1
std_spacing = 0.5  # Reduced standard deviation

mean_friction_angle = 32
std_friction_angle = 5

mean_cohesion = 0.010 # MPa
std_cohesion = 0.00025 # MPa

mean_length = 4.72 # LMU

rock_density = 2700  # kg/m³
gravity = 9.81  # m/s²

correlation_length = 5  # Correlation length set to 10 meters

# Seed the random number generator for consistency (optional)
# np.random.seed(42)

# Number of points (realizations)
N = 256
dx = bench_height / N  # Spatial step size

# Generate correlated fields for spacing and dip
field_spacing = generate_correlated_field(N, dx, correlation_length, mean_spacing, std_spacing)
field_dip = generate_correlated_field(N, dx, correlation_length, mean_dip, std_dip)

# Ensure spacing is positive without altering the mean
field_spacing[field_spacing <= 0] = 0.01  # Set negative spacings to small positive value

# Clip dips to realistic values (5 to 85 degrees to avoid extreme angles)
field_dip = np.clip(field_dip, 5, 85)

# Starting distance calculation
uo = np.random.uniform(0, 1)  # Uniform random number U[0,1]
SMU = mean_spacing
FID = np.zeros((3, N))
FID[0, 0] = uo * SMU  # FID[0,0] - Starting distance from bench toe

# Mean dip DMU
DMU = mean_dip

# Convert angles to radians for trigonometric functions
A_rad = np.radians(dip_VP)
DMU_rad = np.radians(DMU)

cos_A_minus_DMU = np.cos(A_rad - DMU_rad)
if np.abs(cos_A_minus_DMU) < 1e-6:
    cos_A_minus_DMU = 1e-6  # Avoid division by zero

sin_A = np.sin(A_rad)

# Initialize arrays
probabilities = []
x_crest = []
FID_filtered = []
prob_sliding = []  # Probability of sliding for each fracture
prob_L_exceeds_required = []  # Probability that L > required length
prob_stability = []  # Probability of stability for each fracture

# Starting x-coordinate at the bench face
x_start = 0

# Loop through fractures
k = 0
while True:
    if k == 0:
        FID[0, k] = uo * SMU  # Starting distance for the first fracture
    else:
        SPA_j = field_spacing[k - 1]  # Spacing between fractures
        # Calculate FID[0,k]
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

# TO DO  DIP DIR NEEDS TO BE PROBABILISTIC
    # Calculate x-coordinate at bench crest
    angle_diff = np.radians(abs(dip_dir_VP - mean_ddr))  # Convert angle_diff to radians
    x_crest_k = x_start + FID[2, k] * np.cos(dip_rad) * np.cos(angle_diff)  # Correct cosine calculation


    # Check if the fracture exceeds bench width
    if x_crest_k <= bench_width:
        # Fracture intersects bench crest within bench width
        P_L = np.exp(-FID[2, k] / mean_length)
        probabilities.append(P_L)
        x_crest.append(x_crest_k)
        FID_filtered.append([FID[0, k], FID[1, k], FID[2, k]])
        prob_L_exceeds_required.append(P_L)

        # Calculate probability of sliding using PEM
        # Define estimation points for friction angle and cohesion
        phi_mean = mean_friction_angle
        phi_std = std_friction_angle
        c_mean = mean_cohesion
        c_std = std_cohesion

        phi_points = [phi_mean - np.sqrt(3) * phi_std, phi_mean + np.sqrt(3) * phi_std]
        c_points = [c_mean - np.sqrt(3) * c_std, c_mean + np.sqrt(3) * c_std]
        weights = [0.5, 0.5]

        # Calculate area (assuming unit width into the page)
        area = FID[2, k] * 1  # m²

        # Calculate weight of the block
        volume = area * 1  # m³ (assuming unit width)
        W = volume * rock_density * gravity  # N

        # Dip angle
        alpha = FID[1, k]  # degrees
        alpha_rad = dip_rad

        # Initialize lists to store FS values
        FS_values = []

        for i in range(2):  # Loop over phi_points
            for j in range(2):  # Loop over c_points
                phi = phi_points[i]
                c = c_points[j]

                # Calculate FS
                FS = (c * area * 1e6 + W * np.cos(alpha_rad) * np.tan(np.radians(phi))) / (W * np.sin(alpha_rad))
                FS_values.append(FS)

        # Corresponding weights for each FS value
        FS_weights = [0.25] * 4  # Since weights are 0.5 * 0.5 for each combination

        # Calculate mean and variance of FS
        FS_mean = sum(w * fs for w, fs in zip(FS_weights, FS_values))
        FS_variance = sum(w * (fs - FS_mean) ** 2 for w, fs in zip(FS_weights, FS_values))
        FS_std = np.sqrt(FS_variance)

        # Calculate probability of sliding (FS <= 1)
        if FS_std == 0:
            Pf = 1.0 if FS_mean <= 1 else 0.0
        else:
            Pf = norm.cdf((1 - FS_mean) / FS_std)
        prob_sliding.append(Pf)

        # Calculate probability of stability
        P_stability = 1 - (P_L * Pf * POC)
        prob_stability.append(P_stability)
    else:
        # Fracture extends beyond bench width; drop it
        pass

    k += 1

num_fractures = len(probabilities)  # Number of fractures that intersect within bench width

# Convert FID_filtered to numpy array
FID_filtered = np.array(FID_filtered).T  # Transpose to match original shape

probabilities = np.array(probabilities)
x_crest = np.array(x_crest)
prob_sliding = np.array(prob_sliding)
prob_L_exceeds_required = np.array(prob_L_exceeds_required)
prob_stability = np.array(prob_stability)

# Plotting the fractures
plt.figure(figsize=(8, bench_height / bench_width * 8))
for i in range(num_fractures):
    y_start = FID_filtered[0, i]
    dip = FID_filtered[1, i]
    length = FID_filtered[2, i]

    # Calculate x and y coordinates of the fracture line
    x_start = 0  # Starting at the bench face
    dip_rad = np.radians(dip)
    x_end = x_start + length * np.cos(dip_rad)
    y_end = y_start + length * np.sin(dip_rad)

    plt.plot([x_start, x_end], [y_start, y_end], 'r-')

# Draw the bench face
plt.plot([0, 0], [0, bench_height], 'k-', linewidth=2)

plt.xlabel('Bench Width (m)')
plt.ylabel('Bench Height (m)')
plt.title('Simulated Fractures within Bench Width')
plt.xlim(0, bench_width)
plt.ylim(0, bench_height)
plt.grid(True)
plt.show()

# Plot probability that L exceeds required length vs. horizontal position at bench crest
plt.figure()
plt.plot(x_crest, prob_L_exceeds_required, 'bo-')
plt.xlabel('Horizontal Position along Bench Crest (m)')
plt.ylabel('Probability P[L > Required Length]')
plt.title('Probability of Fracture Length Exceeding Required Length')
plt.grid(True)
plt.show()

# Plot probability of sliding vs. horizontal position at bench crest
plt.figure()
plt.plot(x_crest, prob_sliding, 'ro-')
plt.xlabel('Horizontal Position along Bench Crest (m)')
plt.ylabel('Probability of Sliding')
plt.title('Probability of Sliding vs. Position along Bench Crest')
plt.grid(True)
plt.show()

# Plot probability of stability vs. horizontal position at bench crest
plt.figure()
plt.plot(x_crest, prob_stability, 'go-')
plt.xlabel('Horizontal Position along Bench Crest (m)')
plt.xlim(0, )
plt.ylim(0, 1)
plt.ylabel('Probability of Stability')
plt.title('Probability of Stability vs. Position along Bench Crest')
plt.grid(True)
plt.show()

# Optional: Print probabilities and positions
for i in range(num_fractures):
    print(f"Fracture {i+1}: x_crest = {x_crest[i]:.2f} m, "
          f"Required Length = {FID_filtered[2, i]:.2f} m, "
          f"P[L > Required Length] = {prob_L_exceeds_required[i]:.4f}, "
          f"Probability of Sliding = {prob_sliding[i]:.4f}, "
          f"Probability of Stability = {prob_stability[i]:.4f}")


