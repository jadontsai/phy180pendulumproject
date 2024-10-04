import numpy as np
from scipy.signal import find_peaks, savgol_filter  # Import savgol_filter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the damped sinusoidal function (outside find_local_extrema)
def damped_sinusoid(t, theta_0, tau, T, phi_0):
    return theta_0 * np.exp(-t / tau) * np.cos(2 * np.pi * t / T + phi_0)

def find_local_extrema(filepath, prominence=0.5, width=0.5, window_length=11, polyorder=3, deriv_threshold=0.01):
    """
    Finds the local maxima in a quasi-sinusoidal dataset loaded from a text file, 
    along with their corresponding 't' values, and fits a damped sinusoidal 
    function to the maxima.

    Args:
        filepath: Path to the text file containing the data.
                  The file should have two columns separated by whitespace: 't' and 'y'.
        prominence: The prominence threshold for peak detection (not used in this version).
        width: The minimum width of the peaks (not used in this version).
        window_length: The length of the filter window (i.e., the number of coefficients). 
                       Must be an odd integer.
        polyorder: The order of the polynomial used to fit the samples. Must be less than 
                   `window_length`.
        deriv_threshold: The threshold for the derivative to identify significant peaks.

    Returns:
        A tuple containing:
            - t: The array of 't' values.
            - y: The array of 'y' values, centered on zero.
            - local_maxima_t: Corresponding 't' values for the local maxima.
            - local_maxima_y: Corresponding 'y' values for the local maxima, centered on zero.
            - popt: Optimal parameters for the fitted damped sinusoidal function.
            - pcov: Covariance matrix of the fitted parameters.
    """

    # Load data from the text file
    data = np.loadtxt(filepath)
    t = data[:, 0]  # First column is 't'
    y = data[:, 1]  # Second column is 'y'

    # Shift 't' values to start at 0
    t = t - t.min() 
    
    # Center the y values on zero
    y_centered = y - np.mean(y)

    # Smooth the data using a Savitzky-Golay filter
    y_smooth = savgol_filter(y_centered, window_length, polyorder)

    # Calculate the derivative of the smoothed data
    dy_dt = np.gradient(y_smooth, t)

    # Find where the derivative crosses zero from positive to negative
    local_maxima_indices = np.where((dy_dt[:-1] > 0) & (dy_dt[1:] < 0))[0]

    # Refine peak selection using the derivative threshold
    local_maxima_t = []
    local_maxima_y = []
    for i in local_maxima_indices:
        if dy_dt[i] > deriv_threshold:  # Check if the derivative is above the threshold
            local_maxima_t.append(t[i])
            local_maxima_y.append(y_centered[i])

    # Convert to numpy arrays for consistency
    local_maxima_t = np.array(local_maxima_t)
    local_maxima_y = np.array(local_maxima_y)

    # Initial guess for the parameters (adjust these if needed)
    p0 = [max(y_centered), 10, 1, 0]  

    # Fit the damped sinusoidal function to the maxima
    popt, pcov = curve_fit(damped_sinusoid, local_maxima_t, local_maxima_y, 
                          p0=p0, maxfev=10000)

    return t, y_centered, local_maxima_t, local_maxima_y, popt, pcov

def count_cycles_to_decay(local_maxima_y):
    """
    Counts the number of cycles it takes for the amplitude of local maxima 
    to decrease by a factor of 1/e.

    Args:
        local_maxima_y: An array of the y-values of the local maxima.

    Returns:
        The number of cycles for the amplitude to decrease by 1/e.
    """
    initial_amplitude = local_maxima_y[0]
    threshold_amplitude = initial_amplitude / np.e
    cycles = 0
    for i in range(1, len(local_maxima_y)):
        if local_maxima_y[i] < threshold_amplitude:
            cycles = i
            break
    return cycles


# Example usage (remember to adjust the filepath)
filepath = "C:\\Users\\Jadon\\Downloads\\pyhlabtrial.txt"  

t, y_centered, local_maxima_t, local_maxima_y, popt, pcov = find_local_extrema(
    filepath, window_length=11, polyorder=3, deriv_threshold=0.01  # Adjust as needed
)

# Count the cycles to decay
cycles = count_cycles_to_decay(local_maxima_y)
print(f"Number of cycles for amplitude to decrease by 1/e: {cycles:.2f}")

# Plot the data and the fitted curve
plt.figure(figsize=(15, 8))
plt.plot(t, y_centered, 'o-', label='Original Data (Centered)')
plt.plot(local_maxima_t, local_maxima_y, 'ro', label='Local Maxima')

# Generate points for the fitted curve
t_fit = np.linspace(min(local_maxima_t), max(local_maxima_t), 100)
y_fit = damped_sinusoid(t_fit, *popt)  # Use the fitted parameters

plt.plot(t_fit, y_fit, 'r-', label='Fitted Damped Sinusoid')

# Add the equation of the fitted line (adjust formatting as needed)
equation = f'θ(t) = {popt[0]:.2f} * exp(-t / {popt[1]:.2f}) * cos(2πt / {popt[2]:.2f} + {popt[3]:.2f})'
plt.text(0.5, 0.8, equation, transform=plt.gca().transAxes, fontsize=12)

plt.xlabel('t')
plt.ylabel('x')  # Or 'y' if that's what your data represents
plt.title('Local Maxima with Damped Sinusoidal Fit (Centered Data)')
plt.legend()
plt.grid(True)
plt.show()