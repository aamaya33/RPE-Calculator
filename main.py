import serial
import time
import json
import matplotlib.pyplot as plt
from serial.serialutil import SerialException
import numpy as np 
from scipy.signal import find_peaks

def setup_serial(port: str = '/dev/cu.usbmodem1101', baud_rate: int = 9600) -> serial.Serial:
    """Initialize serial connection.
    
    Args:
        port: Serial port name
        baud_rate: Baud rate for serial communication
        
    Returns:
        Serial connection object
    """
    try:
        ser = serial.Serial(port, baud_rate)
        print("Serial connection established")
        time.sleep(2)  # Wait for initialization
        return ser
    except SerialException as e:
        print(f"Error opening serial port: {e}")
        raise

def collect_data(ser: serial.Serial) -> tuple[list[float], list[float]]:
    """Collect time and distance data from serial port.
    
    Args:
        ser: Serial connection object
        
    Returns:
        Tuple of time and distance data lists
    """
    time_data = []
    distance_data = []
    
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                parts = line.split('\t')
                
                if len(parts) == 3:
                    try:
                        elapsed_time = float(parts[0])
                        distance_cm = float(parts[2].replace(" cm", ""))
                        time_data.append(elapsed_time)
                        distance_data.append(distance_cm)
                        print(f"Time: {elapsed_time} s, Distance: {distance_cm} cm")
                    except ValueError:
                        print("Error parsing data")
    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    finally:
        ser.close()
        
    return time_data, distance_data

def find_velocity(distance_data: list[float], time_data: list[float]) -> list[float]:
    velocity_data = []
    for i in range(1, len(distance_data)):
        delta_d = distance_data[i] - distance_data[i-1]
        delta_t = time_data[i] - time_data[i-1]
        velocity = delta_d / delta_t if delta_t != 0 else 0
        velocity_data.append(velocity)
    
    return velocity_data

#MARK: - calculate_rpe
def calculate_rpe(time_data: list[float], distance_data: list[float], velocity_data: list[float]) -> tuple[float, str]:
    """Calculate Rate of Perceived Exertion (RPE) based on lift metrics.
    
    Args:
        time_data: List of time measurements
        distance_data: List of distance measurements
        velocity_data: List of velocity measurements
    
    Returns:
        Tuple of (RPE score, lift type)
    """
    # Detect lift type
    # For bench: Look for U-shape (single minimum)
    # For deadlift: Look for cubic shape (inflection points)
    peaks, _ = find_peaks(distance_data)
    valleys, _ = find_peaks([-x for x in distance_data])
    
    lift_type = "bench" if len(valleys) == 1 else "deadlift"

    if lift_type == "bench":
        # For bench, find the lowest point (valley) and take data after that
        valley_idx = valleys[0]  # We know there's exactly one valley for bench
        ascent_velocity = velocity_data[valley_idx:]
    elif lift_type == "deadlift":
        # For deadlift, take data from start until first peak
        peak_idx = peaks[0]  # First peak
        ascent_velocity = velocity_data[:peak_idx]

    # Calculate average velocity for ascent portion only
    avg_velocity = np.mean(ascent_velocity)
    
    # Calculate metrics
    total_time = time_data[-1] - time_data[0]
    #need to calculate average velocity, but only look at ascent for deads and bench 
    avg_velocity = np.mean(velocity_data)
    
    # Find sticking points (periods of low velocity)
    sticking_threshold = 0.1 * max(velocity_data)  # 10% of max velocity
    sticking_points = sum(1 for v in velocity_data if v > 0 and v < sticking_threshold)
    sticking_ratio = sticking_points / len(velocity_data)
    
    # Base RPE calculation
    ascent_start, ascent_end = find_ascent_startEnd(distance_data, time_data)
    if(lift_type == "bench" and ascent_end - ascent_start > 2):
        rpe = 7.5
    else: 
        rpe = 5.0
    
    if((ascent_end - ascent_start > 2.5 and lift_type == "bench") or (ascent_end - ascent_start > 3.5 and lift_type == "deadlift")):
        rpe = 8.0
    # Time under tension factor
    if total_time > 5.0:  # Longer lifts are harder cm/s
        rpe += min(2.0, (total_time - 5.0))
    
    # Velocity factors
    if avg_velocity < 7.5:  # slow
        rpe += 1.0
    elif avg_velocity > 20:  # Fast/explosive
        rpe -= 1.0
    
    # Sticking point factor
    rpe += sticking_ratio * 1.5
    
    # Adjust based on lift type
    if lift_type == "bench":
        # Bench press typically has more pronounced sticking points
        rpe += sticking_ratio * 0.5
    
    # Clamp final RPE between 1-10
    rpe = max(1.0, min(10.0, rpe))
    
    return round(rpe, 1), lift_type
#MARK: - Plot
def plot_data(time_data: list[float], distance_data: list[float], rpe: float = None, lift_type: str = None) -> None:
    """Plot distance vs time graph with RPE information.
    
    Args:
        time_data: list of time measurements
        distance_data: list of distance measurements
        rpe: Rate of Perceived Exertion (optional)
        lift_type: Type of lift performed (optional)
    """
    plt.close('all')
    ascent_start, ascent_end = find_ascent_startEnd(distance_data, time_data)
    velocity_data = find_velocity(distance_data, time_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(time_data, distance_data)
    if ascent_start and ascent_end:
        ax1.axvline(x=ascent_start, color='r', linestyle='--', label='Ascent Start')
        ax1.axvline(x=ascent_end, color='g', linestyle='--', label='Ascent End')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (cm)')
    ax1.set_title('Distance vs. Time')
    ax1.grid(True)
    
    # Add RPE and lift type info if provided
    if rpe is not None and lift_type is not None:
        info_text = f"Lift Type: {lift_type.capitalize()}\nRPE: {rpe}/10"
        ax1.text(0.95, 0.95, info_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.plot(time_data[1:], velocity_data)
    if ascent_start and ascent_end:
        ax2.axvline(x=ascent_start, color='r', linestyle='--', label='Ascent Start')
        ax2.axvline(x=ascent_end, color='g', linestyle='--', label='Ascent End')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (cm/s)')
    ax2.set_title('Velocity vs. Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.close('all')

def analyze_lift(time_data, distance_data):
    velocity_data = find_velocity(distance_data, time_data)

    rpe, lift_type = calculate_rpe(time_data, distance_data, velocity_data)
    print(f"\nLift Analysis:")
    print(f"Lift Type: {lift_type.capitalize()}")
    print(f"Estimated RPE: {rpe}/10")
    print(f"Total Time: {time_data[-1] - time_data[0]:.2f} s")
    print(f"Average Velocity: {np.mean(velocity_data):.2f} cm/s")
    plot_data(time_data, distance_data, rpe, lift_type)

def save_data(time_data: list[float], distance_data: list[float], filename: str = 'LifterData.json') -> None:
    """Save data to JSON file.
    
    Args:
        time_data: list of time measurements
        distance_data: list of distance measurements
        filename: Output JSON filename
    """
    data = {'time': time_data, 'distance': distance_data}
    try:
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Data saved to {filename}")
    except IOError as e:
        print(f"Error saving data: {e}")

def load_data(filename: str = 'data.json') -> tuple[list[float], list[float]]:
    """Load data from JSON file.
    
    Args:
        filename: Input JSON filename
        
    Returns:
        Tuple of time and distance data lists
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['time'], data['distance']
    except IOError as e:
        print(f"Error loading data: {e}")
        return [], []
#MARK: - Ascend
def find_ascent_startEnd(distance_data, time_data) -> tuple[int, int]:
    # Calculate rate of change
    derivative = np.diff(distance_data)
    velocity_data = find_velocity(distance_data, time_data)
    
    # Find where derivative changes from negative to positive
    peaks, properties = find_peaks(distance_data, height=0, width=1)
    if len(peaks) > 0:
        ascent_start = int(properties["left_bases"][0])
    else:
        for num in velocity_data:
            if num > 0:
                ascent_start = velocity_data.index(num)
                break

    post_ascent_data = distance_data[ascent_start:]
    local_max_idx = post_ascent_data.index(max(post_ascent_data))
    ascent_end = local_max_idx + ascent_start
    
    return time_data[ascent_start], time_data[ascent_end]
#MARK: - Main
def main():
    """Main program execution."""
    try:
        ser = setup_serial()
        time_data, distance_data = collect_data(ser)
        
        if time_data and distance_data:
            save_data(time_data, distance_data)
            analyze_lift(time_data, distance_data)
    except Exception as e:
        print(f"Program error: {e}")
#MARK: - Test
def test_rpe_calculator():
    # Sample data from the table
    time_dataB1 = [
        3.33, 3.37, 3.40, 3.43, 3.47, 3.50, 3.53, 3.57, 3.60, 3.63,
        3.67, 3.70, 3.73, 3.77, 3.80, 3.83, 3.87, 3.90, 3.93, 3.97,
        4.00, 4.03, 4.07, 4.10, 4.13, 4.17, 4.20, 4.23, 4.27, 4.30,
        4.333, 4.367, 4.4, 4.433, 4.467, 4.5, 4.533, 4.567, 4.6,
        4.633, 4.667, 4.7, 4.733, 4.767, 4.8, 4.833, 4.867, 4.9,
        4.933, 4.967, 5.0, 5.033, 5.067, 5.1, 5.133, 5.167, 5.2,
        5.233, 5.267, 5.3, 5.333
    ]
    
    # Convert y values to distance in cm (multiplying by 100 to convert to cm)
    distance_dataB1 = [
        11.3, 11.3, 11.4, 11.6, 11.8, 12.0, 12.2, 12.8, 13.0, 13.3,
        13.6, 13.9, 14.1, 14.3, 14.6, 14.8, 15.0, 15.2, 15.4, 15.7,
        15.8, 16.0, 16.3, 16.5, 16.7, 16.9, 17.0, 17.2, 17.4, 17.6,
        17.9, 18.1, 18.4, 18.7, 18.9, 19.3, 19.6, 19.9, 20.3, 20.6,
        20.9, 21.3, 21.6, 21.8, 22.0, 22.1, 22.1, 22.1, 22.1, 22.1,
        22.2, 22.3, 22.2, 22.2, 22.2, 22.3, 22.3, 22.3, 22.2, 22.2, 
        22.2
    ]

    time_dataD1 = [
        2.13, 2.17, 2.20, 2.23, 2.27, 2.30, 2.33, 2.37, 2.40, 2.43,
        2.47, 2.50, 2.53, 2.57, 2.60, 2.63, 2.67, 2.70, 2.73, 2.77,
        2.80, 2.83, 2.87, 2.90, 2.93, 2.97, 3.00, 3.03, 3.07, 3.10,
        3.13, 3.17, 3.20, 3.23, 3.27, 3.30, 3.33, 3.37, 3.40, 3.43,
        3.47, 3.50, 3.53, 3.57, 3.60, 3.63, 3.67, 3.70, 3.73, 3.77,
        3.80, 3.83, 3.87, 3.90, 3.93, 3.97, 4.00, 4.03, 4.07, 4.10
    ]
    
    distance_dataD1 = [
        2.56, 2.73, 2.95, 3.15, 3.36, 3.58, 3.77, 3.98, 4.05, 4.27,
        4.45, 4.50, 4.69, 4.91, 5.14, 5.39, 5.72, 6.00, 6.48, 7.09,
        7.73, 8.57, 9.48, 10.5, 11.5, 12.7, 14.0, 15.3, 16.8, 18.4,
        20.1, 21.8, 23.7, 25.7, 27.6, 29.8, 31.9, 34.0, 36.2, 38.3,
        40.2, 42.0, 43.7, 45.2, 46.5, 47.6, 48.5, 49.3, 49.9, 50.3,
        50.7, 51.0, 51.2, 51.3, 51.4, 51.5, 51.6, 51.5, 51.5, 51.4
    ]

    time_dataB2 = [
        0.00, 0.03, 0.07, 0.10, 0.13, 0.17, 0.20, 0.23, 0.27, 0.30,
        0.33, 0.37, 0.40, 0.43, 0.47, 0.50, 0.53, 0.57, 0.60, 0.63,
        0.67, 0.70, 0.73, 0.77, 0.80, 0.83, 0.87, 0.90, 0.93, 0.97,
        1.00, 1.03, 1.07, 1.10, 1.13, 1.17, 1.20, 1.23, 1.27, 1.30,
        1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.53, 1.57, 1.60, 1.63,
        1.67, 1.70, 1.73, 1.77, 1.80, 1.83, 1.87, 1.90, 1.93, 1.97,
        2.00, 2.03, 2.07, 2.10, 2.13, 2.17, 2.20, 2.23, 2.27, 2.30,
        2.33, 2.37, 2.40, 2.43, 2.47, 2.50, 2.53, 2.57, 2.60, 2.63,
        2.67, 2.70, 2.73, 2.77, 2.80, 2.83, 2.87, 2.90, 2.93, 2.97,
        3.00, 3.03, 3.07, 3.10, 3.13, 3.17, 3.20, 3.23, 3.27, 3.30,
    ]
    
    distance_dataB2 = [
        24.2, 24.1, 23.9, 23.6, 23.4, 23.1, 22.7, 22.4, 22.0, 21.7,
        21.3, 20.9, 20.6, 20.4, 20.1, 19.9, 19.7, 19.6, 19.4, 19.3,
        19.1, 19.0, 18.8, 18.7, 18.5, 18.4, 18.3, 18.2, 18.1, 18.0,
        17.9, 17.9, 17.8, 17.8, 17.8, 17.8, 17.8, 17.8, 17.8, 17.9,
        18.0, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9,
        19.0, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9,
        20.0, 20.3, 20.6, 20.9, 21.2, 21.5, 21.8, 22.2, 22.5, 22.7,
        23.1, 23.4, 23.7, 23.9, 24.0, 24.1, 24.2, 24.2, 24.2, 24.2,
        24.2, 24.2, 24.2, 24.2, 24.2, 24.2, 24.2, 24.2, 24.2, 24.2,
        24.2, 24.2, 24.2, 24.2, 24.2, 24.2, 24.2, 24.2, 24.2, 24.2
    ]

    time_dataB3 = [
    0.1, 0.1333333, 0.1666667, 0.2, 0.2333333, 0.2666667, 0.3, 0.3333333, 
    0.3666667, 0.4, 0.4333333, 0.4666667, 0.5, 0.5333333, 0.5666667, 0.6, 
    0.6333333, 0.6666667, 0.7, 0.7333333, 0.7666667, 0.8, 0.8333333, 0.8666667, 
    0.9, 0.9333333, 0.9666667, 1.0, 1.033333, 1.066667, 1.1, 1.133333, 1.166667, 
    1.2, 1.233333, 1.266667, 1.3, 1.333333, 1.366667, 1.4, 1.433333, 1.466667, 
    1.5, 1.533333, 1.566667, 1.6, 1.633333, 1.666667, 1.7, 1.733333, 1.766667, 
    1.8, 1.833333, 1.866667, 1.9, 1.933333, 1.966667, 2.0, 2.033333, 2.066667, 
    2.1, 2.133333, 2.166667, 2.2, 2.233333, 2.266667, 2.3, 2.333333, 2.366667, 
    2.4, 2.433333, 2.466667, 2.5, 2.533333, 2.566667, 2.6, 2.633333, 2.666667, 
    2.7, 2.733333, 2.766667, 2.8, 2.833333, 2.866667, 2.9, 2.933333, 2.966667, 
    3.0, 3.033333, 3.066667, 3.1, 3.133333, 3.166667, 3.2, 3.233333, 3.266667, 
    3.3, 3.333333, 3.366667, 3.4, 3.433333, 3.466667, 3.5, 3.533333, 3.566667, 
    3.6, 3.633333, 3.666667, 3.7, 3.733333, 3.766667, 3.8, 3.833333, 3.866667, 
    3.9, 3.933333, 3.966667, 4.0, 4.033333, 4.066667, 4.1, 4.133333, 4.166667, 
    4.2, 4.233333, 4.266667, 4.3, 4.333333, 4.366667, 4.4, 4.433333, 4.466667, 
    4.5, 4.533333, 4.566667
    ]

    distance_dataB3 = [
    30.20563, 30.02424, 29.72241, 29.40901, 29.10087, 28.77413, 28.45023, 28.15177, 
    28.02417, 27.52116, 27.20994, 26.71581, 26.39933, 26.09406, 25.76731, 25.45161, 
    25.14179, 24.82844, 24.50532, 24.18068, 24.03321, 23.89606, 23.57847, 23.41481, 
    23.26987, 23.10843, 22.94287, 22.77903, 22.78817, 22.60433, 22.60496, 22.78898, 
    22.77102, 22.94137, 22.93256, 23.10483, 23.10639, 23.09030, 23.09720, 23.08103, 
    23.25083, 23.24333, 23.24291, 23.41531, 23.39416, 23.56051, 23.55985, 23.72781, 
    23.88472, 24.05248, 24.04835, 24.21514, 24.37473, 24.36873, 24.35716, 24.53164, 
    24.51544, 24.50494, 24.68801, 24.68420, 24.68319, 24.68215, 24.68155, 24.66446, 
    24.85306, 24.85570, 24.84307, 24.83678, 24.83744, 25.01499, 25.01750, 25.18463, 
    25.20387, 25.20404, 25.00184, 25.18334, 25.56200, 25.56280, 25.56465, 25.56719, 
    25.56916, 25.36305, 25.33473, 25.34097, 25.72445, 25.72545, 25.72401, 25.72240, 
    25.72528, 25.72903, 25.49506, 25.86941, 25.87649, 25.67867, 25.65738, 25.82130, 
    25.81360, 26.19548, 25.97218, 26.13398, 26.12665, 26.30350, 26.45919, 26.43601, 
    26.60164, 26.78275, 26.94047, 27.10053, 27.25585, 27.39907, 27.55491, 27.69961, 
    27.86278, 28.18519, 28.33771, 28.48714, 28.82133, 28.96616, 29.29105, 29.42844, 
    30.30489, 30.48150, 30.77478, 31.09117, 31.41284, 31.38241, 31.54033, 31.70732, 
    31.69149, 32.05364, 32.22445, 32.23239, 32.22848, 32.22563, 32.23366
    ]

    time_dataD2 = [
    1.800000, 1.833333, 1.866667, 1.900000, 1.933333, 1.966667, 2.000000, 2.033333,
    2.066667, 2.100000, 2.133333, 2.166667, 2.200000, 2.233333, 2.266667, 2.300000,
    2.333333, 2.366667, 2.400000, 2.433333, 2.466667, 2.500000, 2.533333, 2.566667,
    2.600000, 2.633333, 2.666667, 2.700000, 2.733333, 2.766667, 2.800000, 2.833333,
    2.866667, 2.900000, 2.933333, 2.966667, 3.000000, 3.033333, 3.066667, 3.100000,
    3.133333, 3.166667, 3.200000, 3.233333, 3.266667, 3.300000, 3.333333, 3.366667,
    3.400000, 3.433333, 3.466667, 3.500000, 3.533333, 3.566667, 3.600000, 3.633333,
    3.666667, 3.700000, 3.733333, 3.766667, 3.800000, 3.833333, 3.866667, 3.900000,
    3.933333, 3.966667, 4.000000, 4.033333, 4.066667, 4.100000, 4.133333, 4.166667,
    4.200000, 4.233333, 4.266667, 4.300000, 4.333333, 4.366667, 4.400000, 4.433333,
    4.466667, 4.500000, 4.533333, 4.566667, 4.600000, 4.633333, 4.666667, 4.700000,
    4.733333, 4.766667, 4.800000, 4.833333, 4.866667, 4.900000, 4.933333, 4.966667,
    5.000000, 5.033333, 5.066667, 5.100000, 5.133333, 5.166667, 5.200000, 5.233333,
    5.266667, 5.300000, 5.333333, 5.366667, 5.400000, 5.433333, 5.466667, 5.500000,
    5.533333, 5.566667, 5.600000, 5.633333, 5.666667, 5.700000, 5.733333, 5.766667,
    5.800000, 5.833333, 5.866667, 5.900000, 5.933333, 5.966667, 6.000000, 6.033333,
    6.066667, 6.100000, 6.133333, 6.166667
    ]

    distance_dataD2 = [
    37.53444, 37.58412, 37.63641, 37.69409, 37.80218, 37.83315, 37.82485, 37.81851,
    37.92684, 37.91883, 38.00459, 37.98489, 38.09460, 38.16599, 38.38782, 38.60642,
    38.89927, 39.24522, 39.75954, 40.24832, 40.85042, 41.53195, 42.32357, 43.11903,
    44.11119, 45.01050, 46.14960, 47.26082, 48.42852, 49.68406, 50.97455, 52.28194,
    53.59667, 54.94776, 56.35010, 57.71237, 59.11727, 60.54492, 62.01233, 63.52793,
    64.97621, 66.44989, 67.90216, 69.29507, 70.63019, 71.93189, 73.11055, 74.17171,
    75.17079, 76.04353, 76.83555, 77.50559, 78.10163, 78.57640, 79.01072, 79.35640,
    79.61813, 79.84510, 80.03283, 80.19108, 80.36311, 80.55824, 80.78485, 80.98045,
    81.27886, 81.58757, 81.87265, 82.20803, 82.53431, 82.85905, 83.17527, 83.51173,
    83.89885, 84.30148, 84.73283, 85.11900, 85.54454, 85.93460, 86.29629, 86.56360,
    86.79172, 87.02013, 87.20605, 87.40739, 87.56732, 87.78735, 87.98630, 88.24709,
    88.42887, 88.66557, 88.80153, 88.95102, 89.05860, 89.21913, 89.32325, 89.47548,
    89.58629, 89.78150, 89.92637, 90.06000, 90.13601, 90.18674, 90.22123, 90.25884,
    90.27700, 90.30477, 90.33944, 90.38356, 90.45468, 90.48816, 90.54179, 90.55151,
    90.50869, 90.50403, 90.46765, 90.37704, 90.34524, 90.32132, 90.26236, 90.26172,
    90.26064, 90.27179, 90.36981, 90.37271, 90.43570, 90.53524, 90.54871, 90.43362,
    89.90338, 88.69367, 86.55269, 83.24911
    ]
    analyze_lift(time_dataD2, distance_dataD2)


if __name__ == "__main__":
    test_rpe_calculator()