import numpy as np

# Function to calculate the intersection line of two planes
def plane_intersection(dip1, dip_dir1, dip2, dip_dir2):
    def normal_vector(dip, dip_dir):
        # Convert dip and dip direction to radians
        dip_rad = np.radians(dip)
        dip_dir_rad = np.radians(dip_dir)
        
        # Calculate components of the normal vector
        nx = np.sin(dip_rad) * np.sin(dip_dir_rad)
        ny = np.sin(dip_rad) * np.cos(dip_dir_rad)
        nz = np.cos(dip_rad)
        
        return np.array([nx, ny, nz])
    
    # Calculate normal vectors to the two planes
    n1 = normal_vector(dip1, dip_dir1)
    n2 = normal_vector(dip2, dip_dir2)
    
    # Cross product of the two normal vectors gives the direction of the intersection line
    intersection_line = np.cross(n1, n2)
    
    # Normalize the intersection line vector
    intersection_line /= np.linalg.norm(intersection_line)
    
    # Calculate horizontal component magnitude
    h = np.hypot(intersection_line[0], intersection_line[1])
    
    # Since Z is positive down, we take negative of Z-component for calculations
    z = -intersection_line[2]
    
    # Calculate plunge
    plunge = np.degrees(np.arctan2(z, h))
    if plunge < 0:
        plunge += 360  # Ensure plunge is within [0, 360) degrees
    
    # Calculate trend (azimuth)
    trend = (np.degrees(np.arctan2(intersection_line[0], intersection_line[1])) + 360) % 360
    
    # Adjust trend and plunge if the line is plunging upward
    if intersection_line[2] > 0:  # Plunging upward
        trend = (trend + 180) % 360
        plunge = -plunge  # Negative plunge indicates upward plunge
    
    return plunge, trend

# Input values for Set 1 and Set 2
dip1 = 50  # Dip of Joint Set 1
dip_dir1 = 105  # Dip direction of Joint Set 1
dip2 = 65  # Dip of Joint Set 2
dip_dir2 = 235  # Dip direction of Joint Set 2

# Call the function to compute plunge and trend
plunge, trend = plane_intersection(dip1, dip_dir1, dip2, dip_dir2)

# Output the results
print(f"Plunge of the intersection line: {plunge:.1f} degrees")
print(f"Trend of the intersection line: {trend:.1f} degrees")
