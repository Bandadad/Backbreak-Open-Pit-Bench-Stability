def compute_factor_of_safety(N1, N2, W, a, b, c1, c2, phi1, phi2, A1, A2, nu, i, 
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
    FOS = compute_original_fos(N1, N2, W, a, b, c1, c2, phi1, phi2, A1, A2, nu, i)

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