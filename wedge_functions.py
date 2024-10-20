import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def compute_factor_of_safety(N1, N2, W, a, b, A1, A2, nu, i, 
                             c1_mean, c1_std, c2_mean, c2_std, phi1_mean, phi1_std, phi2_mean, phi2_std):
    """
    Returns:
        FOS: Factor of Safety
        prob_failure: Probability of Failure (when FOS < 1.0)
    """

    # Define failure as Factor of Safety < 1.0
    def point_estimate(mean, std, N):
        """Helper function to generate point estimate samples."""
        return np.random.normal(mean, std, N)

    # Generate sample values using the point estimate method
    samples = 1000  # You can adjust the sample size as needed
    phi1_samples = point_estimate(phi1_mean, phi1_std, samples)
    phi2_samples = point_estimate(phi2_mean, phi2_std, samples)
    c1_samples = point_estimate(c1_mean, c1_std, samples)
    c2_samples = point_estimate(c2_mean, c2_std, samples)

    fos_samples = []

    # Process each of the three cases separately
    for p1, p2, coh1, coh2 in zip(phi1_samples, phi2_samples, c1_samples, c2_samples):
        
        # Case 1: N1 < 0 and N2 < 0 (no shear strength, immediate failure)
        if N1 < 0 and N2 < 0:
            fos_samples.append(0)  # FOS = 0
        
        # Case 2: N1 > 0 and N2 < 0 (compute FOS for plane 1)
        elif N1 > 0 and N2 < 0:
            N_a = W * a[2]
            S_a = np.sqrt((N_a * a[0])**2 + (N_a * a[1])**2 + (N_a * a[2] + W)**2)
            Q_a = N_a * np.tan(np.radians(p1)) + coh1 * A1
            FOS_1 = Q_a / S_a
            fos_samples.append(FOS_1)
        
        # Case 3: N1 < 0 and N2 > 0 (compute FOS for plane 2)
        elif N1 < 0 and N2 > 0:
            N_b = W * b[2]
            S_b = np.sqrt((N_b * b[0])**2 + (N_b * b[1])**2 + (N_b * b[2] + W)**2)
            Q_b = N_b * np.tan(np.radians(p2)) + coh2 * A2
            FOS_2 = Q_b / S_b
            fos_samples.append(FOS_2)
        
        # Case 4: N1 > 0 and N2 > 0 (compute FOS for both planes)
        elif N1 > 0 and N2 > 0:
            S = nu * W * i[2]
            Q = N1 * np.tan(np.radians(p1)) + N2 * np.tan(np.radians(p2)) + coh1 * A1 + coh2 * A2
            FOS_3 = Q / S
            fos_samples.append(FOS_3)

    # Calculate probability of failure
    failure_count = np.sum(np.array(fos_samples) < 1.0)
    probability_of_failure = failure_count / samples

    # Calculate the FOS based on the original (deterministic) parameters
    FOS = compute_original_fos(N1, N2, W, a, b, c1_mean, c2_mean, phi1_mean, phi2_mean, A1, A2, nu, i)

    return FOS, probability_of_failure

def compute_original_fos(N1, N2, W, a, b, c1, c2, phi1, phi2, A1, A2, nu, i):
    """Helper function to calculate the original deterministic Factor of Safety."""
    if N1 < 0 and N2 < 0:
        return 0
    elif N1 > 0 and N2 < 0:
        N_a = W * a[2]
        S_a = np.sqrt((N_a * a[0])**2 + (N_a * a[1])**2 + (N_a * a[2] + W)**2)
        Q_a = N_a * np.tan(np.radians(phi1)) + c1 * A1
        return Q_a / S_a
    elif N1 < 0 and N2 > 0:
        N_b = W * b[2]
        S_b = np.sqrt((N_b * b[0])**2 + (N_b * b[1])**2 + (N_b * b[2] + W)**2)
        Q_b = N_b * np.tan(np.radians(phi2)) + c2 * A2
        return Q_b / S_b
    elif N1 > 0 and N2 > 0:
        S = nu * W * i[2]
        Q = N1 * np.tan(np.radians(phi1)) + N2 * np.tan(np.radians(phi2)) + c1 * A1 + c2 * A2
        return Q / S


def distance_from_crest(h, plunge, trend, alpha4):
    # Convert plunge and alpha4 to radians
    plunge_rad = np.radians(plunge)
    trend_rad = np.radians(trend)
    alpha4_rad = np.radians(alpha4)
    
    cot_plunge = 1 / np.tan(plunge_rad)
    angle_diff = trend_rad - alpha4_rad
    
    distance = h * cot_plunge * np.cos(angle_diff)
    
    return distance


def distance_to_tension_crack(L, alpha1, alpha4):
    alpha1_rad = np.radians(alpha1)
    alpha4_rad = np.radians(alpha4)
    angle_diff = alpha4_rad - alpha1_rad
    distance = L * np.sin(angle_diff)

    return distance


# Function to compute the unit normal vector of a plane
def unit_normal_vector(dip, alpha):
    dip_rad = np.radians(dip)  
    alpha_rad = np.radians(alpha)  
    nx = np.sin(dip_rad) * np.sin(alpha_rad)  
    ny = np.sin(dip_rad) * np.cos(alpha_rad)  
    nz = np.cos(dip_rad)  

    return np.array([nx, ny, nz])


# Cross products to find lines of intersection (using the right-hand rule)
def line_of_intersection(n1, n2):
    intersection_vector = np.cross(n1, n2)  # Cross product of two normal vectors

    return intersection_vector


def process_wedges(dip1, alpha1, c1_mean, c1_std, phi1_mean, phi1_std, dip2, alpha2, c2_mean, c2_std, phi2_mean, phi2_std, H1, alpha4, gamma):
    # Input angles and geometry (from the example in the PDF)
    dip3 = 0   # Dip of plane 3 (degrees)
    dip4 = 90  # Dip of slope face (degrees)
    dip5 = 90  # Dip of tension crack (degrees)
    alpha3 = alpha4  # Dip direction of plane 3 (degrees)
    alpha5 = alpha4  # Dip direction of plane 5 (degrees)

    eta = 1   # +1 since tension crack does not have an overhang
    
    # Calculate the unit normal vectors for all planes
    a = unit_normal_vector(dip1, alpha1)  # Plane 1 (normal vector a)
    b = unit_normal_vector(dip2, alpha2)  # Plane 2 (normal vector b)
    d = unit_normal_vector(dip3, alpha3)  # Plane 3 (upper slope surface, normal vector d)
    f = unit_normal_vector(dip4, alpha4)  # Plane 4 (slope face, normal vector f)
    f5 = unit_normal_vector(dip5, alpha5) # Plane 5 (tension crack, normal vector f5)

    # Calculate the components of the intersection vectors using PDF notation
    g = -line_of_intersection(a, f)    # Intersection of plane 1 and plane 4
    g5 = -line_of_intersection(a, f5)  # Intersection of plane 1 and plane 5
    i = -line_of_intersection(a, b)    # Intersection of plane 1 and plane 2
    j = -line_of_intersection(d, f)    # Intersection of planes 3 and 4
    j5 = -line_of_intersection(d, f5)  # Intersection of planes 3 and 5
    k =  line_of_intersection(i, b)    # No negative sign since i already assigned negative
    l =  line_of_intersection(a, i)    # No negative sign since i already assigned negative

    # Dot products
    m = np.dot(g, d)    
    m5 = np.dot(g5, d)  
    n = np.dot(b, j)    
    n5 = np.dot(b, j5)  
    p = np.dot(i, d)    
    q = np.dot(b, g)    
    q5 = np.dot(b, g5)  
    r = np.dot(a, b)    
    s5 = np.dot(a, f5)  
    v5 = np.dot(b, f5)  
    w5 = np.dot(i, f5)  
    lambda_val = np.dot(i, g)   
    lambda_5_val = np.dot(i, g5) 
    epsilon = np.dot(f, f5)  

    # Miscellaneous factors 
    R = np.sqrt(1 - r**2)  
    rho = (1 / R**2) * (n * q / abs(n * q))  
    mu = (1 / R**2) * (m * q / abs(m * q))  
    nu = (1 / R) * (p / abs(p))  
    G = np.dot(g, g)  
    G5 = np.dot(g5, g5)  
    M = np.sqrt(G * p**2 - 2 * m * p * lambda_val + m**2 * R**2) 
    M5 = np.sqrt(G5 * p**2 - 2 * m5 * p * lambda_5_val + m5**2 * R**2) 
    h = H1 / abs(g[2])  
    h5 = 0
    L = (M * h / abs(p)) 
    B = (np.tan(np.radians(phi1_mean))**2 + np.tan(np.radians(phi2_mean))**2 - 2 * (mu * r / rho) * np.tan(np.radians(phi1_mean)) * np.tan(np.radians(phi2_mean))) / R**2  # Equation III.45

    # Calculate Plunge and trend of line of intersection (Equations III.46 - III.47)
    plunge_i = np.degrees(np.arcsin(nu * i[2]))  # Equation III.46
    trend_i = np.degrees(np.arctan2(-nu * i[0], -nu * i[1]))  # Equation III.47
    trend_i = (trend_i + 360) % 360

    # Check on wedge geometry 
    # Wedge formation checks
    if p * i[2] < 0 or n * q * i[2] < 0:
        return pd.Series([trend_i, plunge_i, np.nan, 9999, 1.0, "No wedge formed"])

    # Tension crack checks
    if epsilon * eta * q5 * i[2] < 0 or h5 < 0 or abs(m5 * h5 / (m * h)) > 1 or abs(n * q5 * m5 * h5 / (n5 * q * m * h)) > 1:
        return pd.Series([trend_i, plunge_i, np.nan, 9999, 1.0, "Tension Crack is invalid"])

    # Calculate areas of faces and weight of wedge
    A1 = abs(m * q * h**2 - m5 * q5 * h5**2) / (2 * abs(p))  
    A2 = abs((q * m**2 * h**2) / abs(n) - (q5 * m5**2 * h5**2) / abs(n5)) / (2 * abs(p)) 
    A5 = abs(m5 * q5 * h5**2) / (2 * abs(n5))  
    W = gamma * (q**2 * m**2 * h**3 / abs(n) - q5**2 * m5**2 * h5**3 / abs(n5)) / (6 * abs(p))  

    # Calculate normal reactions on planes 1 and 2 
    N1 = rho * W * k[2]  
    N2 = mu * W * l[2] 

    distance = distance_from_crest(H1, plunge_i, trend_i, alpha4)

    distance_c = distance_to_tension_crack(L, alpha1, alpha4)
    FOS, probability_of_failure = compute_factor_of_safety(N1, N2, W, a, b, A1, A2, nu, i, c1_mean, c1_std, c2_mean, c2_std, phi1_mean, phi1_std, phi2_mean, phi2_std)
 
    return pd.Series([trend_i, plunge_i, distance, FOS, probability_of_failure, "Wedge is formed and tension crack is valid"])