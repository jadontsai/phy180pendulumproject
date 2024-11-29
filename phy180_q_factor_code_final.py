import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

def damped_exp_fixed_tau(t, tau):
    return 0.518 * np.exp(-t / tau)

def find_local_extrema(filepath, window_length=11, polyorder=3, deriv_threshold=0.01):
    """
    Finds the local maxima in a quasi-sinusoidal dataset loaded from a text file, 
    along with their corresponding 't' values, and fits a damped sinusoidal 
    function to the maxima.

    Args:
        filepath: Path to the text file containing the data.
                The file should have two columns separated by whitespace: 't' and 'y'.
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

    with open(filepath, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            line = line.replace('\t', ' ')  # Replace tabs with spaces
            data.append([float(val) for val in line.split()])

    data = np.array(data)
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

    # --- More distinct variable names for curve fitting ---
    maxima_time = np.array(local_maxima_t) 
    maxima_y = np.array(local_maxima_y)

    # 1. Define the function to fit (with fixed 0.518 factor)
    def func(t, tau):
        return 0.518 * np.exp(-t / tau)

    # --- Improved curve fitting ---
    # Estimate an initial guess for tau (adjust this based on visual inspection)
    initial_tau_guess = 5  

    # Bounds for tau (adjust these to constrain the possible values of tau)
    lower_bound = 1  
    upper_bound = 100  

    # Perform the curve fit with initial guess and bounds
    popt, pcov = curve_fit(func, maxima_time, maxima_y, 
                           p0=[initial_tau_guess], 
                           bounds=([lower_bound], [upper_bound]))
    # --- End of improved curve fitting ---

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
filepath = "C:\\Users\\Jadon\\Downloads\\123.txt"  # Replace with your actual file path

t, y_centered, local_maxima_t, local_maxima_y, popt, pcov = find_local_extrema(
    filepath, window_length=11, polyorder=3, deriv_threshold=0.01  # Adjust as needed
)

# Count the cycles to decay
cycles = count_cycles_to_decay(local_maxima_y)
print(f"Number of cycles for amplitude to decrease by 1/e: {cycles:.2f}")

# --- Calculate error bars ---
perr = np.sqrt(np.diag(pcov))  # Standard deviation errors on the parameters
# Assuming you want to show one standard deviation error
y_fit_err = damped_exp_fixed_tau(np.array(local_maxima_t), *(popt + perr)) - damped_exp_fixed_tau(np.array(local_maxima_t), *popt)  
# --- End of error bar calculation ---

# Plot the data and the fitted curve
plt.figure(figsize=(36, 8))
#plt.plot(t, y_centered, 'o-', label='Original Data (Centered)')
plt.plot(local_maxima_t, local_maxima_y, 'yo', label='Local Maxima')

# Generate points for the fitted curve using damped_exp_fixed_tau with ONLY the maxima
t_fit = np.linspace(min(local_maxima_t), max(local_maxima_t), 100)  # Use t values of maxima
y_fit = damped_exp_fixed_tau(t_fit, *popt)  # Use the correct function

plt.plot(t_fit, y_fit, 'r-', label='Exponential Fit')  # Plot the fitted curve

# --- Add error bars to the plot ---
plt.errorbar(local_maxima_t, local_maxima_y, xerr = 0.03, yerr=0.005, fmt='none', ecolor='black', capsize=5)
# --- End of adding error bars ---

# Add the equation of the fitted line (adjust if needed)
equation = f'y = 0.518 * exp(-t / 52.13)' 
plt.text(0.5, 0.8, equation, transform=plt.gca().transAxes, fontsize=38)

plt.xlabel('Time (Seconds)', fontsize=30)
plt.ylabel('Amplitude (Radians)', fontsize=30)  # Or 'y' if that's what your data represents
#plt.title('Local Maxima with Damped Sinusoidal Fit (Centered Data)')

# Increase legend font size
plt.legend(fontsize=25)  # Adjust the font size as needed

plt.grid(True)
plt.show()