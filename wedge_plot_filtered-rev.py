import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wedge_functions import process_wedges


# Function to calculate wedge parameters and factor of safety
def calculate_wedge_params(row):
    # Extract values from the row
    dip1 = row['dip_JP1']
    dip2 = row['dip_JP2']
    alpha1 = row['dip_dir_JP1']
    alpha2 = row['dip_dir_JP2']
    H1 = row['Wedge Height']
    
    # Call the process_wedges function with the extracted parameters and global constants
    trend_i, plunge_i, distance_from_crest, FOS = process_wedges(dip1, alpha1, c1, phi1, dip2, alpha2, c2, phi2, H1, dip_dir_VP)  

    # Return the calculated values for trend, plunge, distance from crest, and FOS
    return pd.Series([trend_i, plunge_i, distance_from_crest, FOS])


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

# Define the vertical plane (VP)
dip_VP = 90
dip_dir_VP = 185
n_VP = compute_normal(dip_VP, dip_dir_VP)
d_VP = 0  # Passing through origin

# Define the vertical plane rectangle in its local coordinate system
width, height = 100, 100
u1 = np.cross(n_VP, np.array([0, 0, 1]))
u1 /= np.linalg.norm(u1)
u2 = np.cross(u1, n_VP)
u2 /= np.linalg.norm(u2)

# Probabilistic sampling parameters for Joint Sets
# Joint Set 1
dip_JP1_mean, dip_JP1_std = 45, 2
dip_dir_JP1_mean, dip_dir_JP1_std = 105, 5
spacing_JP1 = 7.5
phi1 = 20
c1 = 500

# Joint Set 2
dip_JP2_mean, dip_JP2_std = 70, 2
dip_dir_JP2_mean, dip_dir_JP2_std = 235, 5
spacing_JP2 = 5
phi2 = 30
c2 = 1000

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

# Generate planes for Joint Set 1
count_JP1 = int(width / spacing_JP1)
num_planes_JP1 = 2 * count_JP1 + 1
dips_JP1 = np.random.normal(dip_JP1_mean, dip_JP1_std, size=num_planes_JP1)
dip_dirs_JP1 = np.random.normal(dip_dir_JP1_mean, dip_dir_JP1_std, size=num_planes_JP1)
normals_JP1 = [compute_normal(dip, dip_dir) for dip, dip_dir in zip(dips_JP1, dip_dirs_JP1)]
planes_JP1 = generate_planes(normals_JP1, spacing_JP1, count_JP1)

# Generate planes for Joint Set 2
count_JP2 = int(width / spacing_JP2)
num_planes_JP2 = 2 * count_JP2 + 1
dips_JP2 = np.random.normal(dip_JP2_mean, dip_JP2_std, size=num_planes_JP2)
dip_dirs_JP2 = np.random.normal(dip_dir_JP2_mean, dip_dir_JP2_std, size=num_planes_JP2)
normals_JP2 = [compute_normal(dip, dip_dir) for dip, dip_dir in zip(dips_JP2, dip_dirs_JP2)]
planes_JP2 = generate_planes(normals_JP2, spacing_JP2, count_JP2)

# Plotting setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-width/2, width/2)
ax.set_ylim(-height/2, height/2)
ax.set_xlabel('X-axis (ft)')
ax.set_ylabel('Y-axis (ft)')
ax.set_title('Projection of Joint Sets on Vertical Plane\n(Only lines intersecting the top edge are shown)')

rect_corners = np.array([[-width/2, -height/2], [width/2, -height/2], [width/2, height/2], [-width/2, height/2], [-width/2, -height/2]])
ax.plot(rect_corners[:, 0], rect_corners[:, 1], 'k-')

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

# Initialize a DataFrame to store intersection points
intersection_points = []

# Store the filtered lines
filtered_lines_JP1 = []
filtered_lines_JP2 = []

# Plot joint set 1 (blue lines) and filter
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
    tau_s1 = (-width/2 - s0_1) / ds1 if ds1 != 0 else -np.inf
    tau_s2 = (width/2 - s0_1) / ds1 if ds1 != 0 else np.inf
    tau_t1 = (-height/2 - t0_1) / dt1 if dt1 != 0 else -np.inf
    tau_t2 = (height/2 - t0_1) / dt1 if dt1 != 0 else np.inf
    tau_min = max(min(tau_s1, tau_s2), min(tau_t1, tau_t2))
    tau_max = min(max(tau_s1, tau_s2), max(tau_t1, tau_t2))
    if tau_min > tau_max:
        continue
    if not line_intersects_top_edge(s0_1, t0_1, ds1, dt1, tau_min, tau_max, width, height):
        continue
    s1 = s0_1 + tau_min * ds1
    t1 = t0_1 + tau_min * dt1
    s2 = s0_1 + tau_max * ds1
    t2 = t0_1 + tau_max * dt1
    ax.plot([s1, s2], [t1, t2], 'b-')
    # Store the filtered line along with dip and dip_dir
    filtered_lines_JP1.append({'s0': s0_1, 't0': t0_1, 'ds': ds1, 'dt': dt1, 'dip': dip_JP1, 'dip_dir': dip_dir_JP1})

# Plot joint set 2 (red lines), filter, and calculate intersections
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
    tau_s1_2 = (-width/2 - s0_2) / ds2 if ds2 != 0 else -np.inf
    tau_s2_2 = (width/2 - s0_2) / ds2 if ds2 != 0 else np.inf
    tau_t1_2 = (-height/2 - t0_2) / dt2 if dt2 != 0 else -np.inf
    tau_t2_2 = (height/2 - t0_2) / dt2 if dt2 != 0 else np.inf
    tau_min2 = max(min(tau_s1_2, tau_s2_2), min(tau_t1_2, tau_t2_2))
    tau_max2 = min(max(tau_s1_2, tau_s2_2), max(tau_t1_2, tau_t2_2))
    if tau_min2 > tau_max2:
        continue
    if not line_intersects_top_edge(s0_2, t0_2, ds2, dt2, tau_min2, tau_max2, width, height):
        continue
    s1_2 = s0_2 + tau_min2 * ds2
    t1_2 = t0_2 + tau_min2 * dt2
    s2_2 = s0_2 + tau_max2 * ds2
    t2_2 = t0_2 + tau_max2 * dt2
    ax.plot([s1_2, s2_2], [t1_2, t2_2], 'r-')
    # Store the filtered line along with dip and dip_dir
    filtered_lines_JP2.append({'s0': s0_2, 't0': t0_2, 'ds': ds2, 'dt': dt2, 'dip': dip_JP2, 'dip_dir': dip_dir_JP2})

    # Now compute intersections between filtered joint set 1 and this line
    for line_JP1 in filtered_lines_JP1:
        s0_1 = line_JP1['s0']
        t0_1 = line_JP1['t0']
        ds1 = line_JP1['ds']
        dt1 = line_JP1['dt']
        intersection = compute_line_intersection(s0_1, t0_1, ds1, dt1, s0_2, t0_2, ds2, dt2)
        if intersection is not None:
            x, y = intersection
            # Filter by window bounds
            if -width/2 <= x <= width/2 and -height/2 <= y <= height/2:
                # Store intersection point along with dips and dip_dirs of both planes
                intersection_points.append({
                    'X (ft)': x,
                    'Y (ft)': y,
                    'dip_JP1': line_JP1['dip'],
                    'dip_dir_JP1': line_JP1['dip_dir'],
                    'dip_JP2': dip_JP2,
                    'dip_dir_JP2': dip_dir_JP2
                })

# Convert to pandas DataFrame and display
df_intersections = pd.DataFrame(intersection_points)
df_intersections["Wedge Height"] = (height / 2) - df_intersections["Y (ft)"]
df_intersections[['Trend', 'Plunge', 'Distance from Crest', 'Factor of Safety']] = df_intersections.apply(calculate_wedge_params, axis=1)
print(df_intersections)
plt.grid(True)

plt.figure(figsize=(8, 6))
plt.hist(df_intersections['Factor of Safety'], bins=50, color='blue', edgecolor='black')
plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='FOS = 1')
plt.title('Histogram of Factor of Safety Values')
plt.xlabel('Factor of Safety')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()