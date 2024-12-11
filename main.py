import os 
import serial
import time
import json
import matplotlib.pyplot as plt
from serial.serialutil import SerialException
import numpy as np 
from scipy.signal import find_peaks
from MQTT import MQTTHandler

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
    """Find velocity given the distance and time measurements.
    
    Args:
        distance_data: List of distance measurements
        velocity_data: List of velocity measurements
        
    Returns:
        List of velocity points
    """
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
    
    if len(valleys) == 0 and len(peaks) == 0:
        # If no clear peaks/valleys, use simple threshold
        lift_type = "bench" if distance_data[0] > min(distance_data) else "deadlift"
        if lift_type == "bench":
            valley_idx = np.argmin(distance_data)
        else:
            peak_idx = np.argmax(distance_data)
    else:
        lift_type = "bench" if len(valleys) == 1 or distance_data[0] > min(distance_data) else "deadlift"


    if lift_type == "bench":
        # For bench, find the lowest point (valley) and take data after that
        valley_idx = valleys[0] if len(valleys) > 0 else np.argmin(distance_data) # We know there's exactly one valley for bench
        ascent_velocity = velocity_data[valley_idx:]
    elif lift_type == "deadlift":
        # For deadlift, take data from start until first peak
        peak_idx = peaks[0] if len(peaks) > 0 else np.argmax(distance_data)  # First peak
        ascent_velocity = velocity_data[:peak_idx]

    # Calculate average velocity for ascent portion only
    avg_velocity = np.mean(ascent_velocity)
    
    # Calculate metrics
    total_time = time_data[-1] - time_data[0]
    
    # Find sticking points (periods of low velocity)
    sticking_threshold = 0.1 * max(velocity_data)  # 10% of max velocity
    sticking_points = sum(1 for v in velocity_data if v > 0 and v < sticking_threshold)
    sticking_ratio = sticking_points / len(velocity_data)
    
    # Base RPE calculation
    ascent_start, ascent_end = find_ascent_startEnd(distance_data, time_data)
    if(lift_type == "bench" and ascent_end - ascent_start > 2 or lift_type == "deadlift" and ascent_end - ascent_start > 2.5):
        rpe = 7.5
    else: 
        rpe = 5.0
    
    if((ascent_end - ascent_start > 3 and lift_type == "bench") or (ascent_end - ascent_start > 3.75 and lift_type == "deadlift")):
        rpe = 8.0
    # Time under tension factor
    if total_time > 7.0:  # Longer lifts are harder cm/s
        rpe += min(1.0, (total_time - 5.0))
    
    # Velocity factors
    if avg_velocity < 7.5:  # slow
        rpe += 1.0
    elif avg_velocity > 23:  # Fast/explosive
        rpe -= 1.0
    # Noticed that avg velocity for deadlift is better, threshold is 17.5 for another 0.5 rpe 
    if lift_type == "deadlift" and avg_velocity > 17.5:
        rpe += 0.5
    
    # Sticking point factor
    rpe += sticking_ratio * 1.15
    
    # Adjust based on lift type
    if lift_type == "bench":
        # Bench press typically has more pronounced sticking points
        rpe += sticking_ratio * 0.5
    
    # Clamp final RPE between 1-10
    rpe = max(1.0, min(10.0, round(rpe*2)/2))
    
    return round(rpe, 1), lift_type, avg_velocity, ascent_start, ascent_end
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
    try:
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
    except Exception as e:
        print(f"Error plotting data: {e}")
    finally: 
        plt.close('all')

def analyze_lift(time_data, distance_data, rpe_threshold):
    """Analyze and plot lift data. 

    Args:
        time_data: list of time measurements
        distance_data: list of distance measurements
        rpe_threshold: RPE threshold for lift analysis
    
    """
    velocity_data = find_velocity(distance_data, time_data)
    mqtt_handler = MQTTHandler()
    

    rpe, lift_type, average_velocity, ascent_start, ascent_end = calculate_rpe(time_data, distance_data, velocity_data)
    try:
        if not mqtt_handler.connected:
            mqtt_handler.start()
            time.sleep(1)  # Give time for arduino to connect
            
        if mqtt_handler.connected:
            if rpe > rpe_threshold:
                print("RPE is too high, activating servo")
                mqtt_handler.publish("servoMotor", "HIGH")
            else:
                mqtt_handler.publish("servoMotor", "LOW")
        else:
            print("Cannot control servo - MQTT not connected")
            
    except Exception as e:
        print(f"Error controlling servo: {e}")
    mqtt_handler.stop()
    print(f"\nLift Analysis:")
    print(f"Lift Type: {lift_type.capitalize()}")
    print(f"Estimated RPE: {rpe}/10")
    print(f"Total Time: {time_data[-1] - time_data[0]:.2f} s")
    print(f"Average Velocity: {average_velocity:.2f} cm/s")
    print(f"Time under tension: {ascent_end - ascent_start:.2f} s")
    plot_data(time_data, distance_data, rpe, lift_type)

def save_data(time_data: list[float], distance_data: list[float], filename: str = 'LifterData.json') -> None:
    """Save data to JSON file.
    
    Args:
        time_data: list of time measurements
        distance_data: list of distance measurements
        filename: Output JSON filename
    """
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    filepath = os.path.join(desktop_path, filename)

    data = {'time': time_data, 'distance': distance_data}
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Data saved to {filepath}")
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
    """ Finds where in the lift the barbell begins to go up (the ascent portion of the lift)
    
        Args:
            time_data: List of time measurements
            distance_data: List of distance measurements
            velocity_data: List of velocity measurements
        
        Returns:
            Tuple of (RPE score, lift type)
    """
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

def check_acceleration_stop(distance_data, time_window=3.0):
    """Check if acceleration has been ~0 for given time window."""
    if len(distance_data) < 30:  # Need enough data points
        return False
        
    # Calculate acceleration from distance data
    acceleration = np.diff(np.diff(distance_data[-30:]))
    
    # Check if acceleration near zero for last 3 seconds
    threshold = 0.001  # Adjust based on your sensor sensitivity
    return all(abs(a) < threshold for a in acceleration[-int(30 * time_window/3):])
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

    time_dataB4 = [
    9.433333, 9.466667, 9.500000, 9.533333, 9.566667, 9.600000, 9.633333, 9.666667,
    9.700000, 9.733333, 9.766667, 9.800000, 9.833333, 9.866667, 9.900000, 9.933333,
    9.966667, 10.000000, 10.033333, 10.066667, 10.100000, 10.133333, 10.166667, 10.200000,
    10.233333, 10.266667, 10.300000, 10.333333, 10.366667, 10.400000, 10.433333, 10.466667,
    10.500000, 10.533333, 10.566667, 10.600000, 10.633333, 10.666667, 10.700000, 10.733333,
    10.766667, 10.800000, 10.833333, 10.866667, 10.900000, 10.933333, 10.966667, 11.000000,
    11.033333, 11.066667, 11.100000, 11.133333, 11.166667, 11.200000, 11.233333, 11.266667,
    11.300000, 11.333333, 11.366667, 11.400000, 11.433333, 11.466667, 11.500000, 11.533333,
    11.566667, 11.600000, 11.633333, 11.666667, 11.700000, 11.733333, 11.766667, 11.800000,
    11.833333, 11.866667, 11.900000, 11.933333, 11.966667, 12.000000, 12.033333, 12.066667,
    12.100000, 12.133333, 12.166667, 12.200000, 12.233333, 12.266667, 12.300000, 12.333333,
    12.366667, 12.400000, 12.433333, 12.466667, 12.500000, 12.533333, 12.566667, 12.600000,
    12.633333, 12.666667, 12.700000, 12.733333, 12.766667, 12.800000, 12.833333, 12.866667,
    12.900000, 12.933333, 12.966667, 13.000000, 13.033333, 13.066667, 13.100000, 13.133333,
    13.166667, 13.200000, 13.233333, 13.266667, 13.300000, 13.333333, 13.366667, 13.400000,
    13.433333, 13.466667, 13.500000, 13.533333, 13.566667, 13.600000, 13.633333, 13.666667,
    13.700000, 13.733333, 13.766667, 13.800000, 13.833333, 13.866667, 13.900000, 13.933333,
    13.966667, 14.000000, 14.033333, 14.066667, 14.100000, 14.133333, 14.166667, 14.200000,
    14.233333, 14.266667, 14.300000, 14.333333, 14.366667, 14.400000, 14.433333, 14.466667,
    14.500000, 14.533333, 14.566667, 14.600000
    ]

    distance_dataB4 = [
    98.45000, 98.28552, 98.11344, 98.04734, 97.95734, 97.94272, 97.93484, 97.94889,
    97.93901, 98.06695, 98.01151, 97.99786, 97.83473, 97.74190, 97.71493, 97.82161,
    97.89549, 98.01053, 98.10935, 98.24388, 98.27144, 98.22903, 98.13374, 98.06597,
    97.99176, 97.93344, 97.89666, 97.92315, 97.94555, 97.94112, 97.90426, 97.85155,
    97.76819, 97.58635, 97.40967, 97.18759, 96.91254, 96.59550, 95.96458, 95.03102,
    93.50938, 91.66972, 89.58095, 86.92328, 84.16757, 81.03350, 77.85210, 74.69326,
    71.33591, 68.08329, 64.75830, 61.38540, 58.16718, 54.89820, 52.15188, 49.23046,
    46.65529, 44.23495, 41.91164, 39.79274, 37.74440, 35.99382, 34.45205, 33.21221,
    32.22528, 31.53183, 31.20744, 30.92941, 30.97896, 31.01503, 31.15840, 31.27350,
    31.39801, 31.41809, 31.43934, 31.39188, 31.41722, 31.37738, 31.41799, 31.51578,
    31.59807, 31.75206, 31.92953, 32.27223, 32.76462, 33.47711, 34.33508, 35.28136,
    36.31034, 37.52542, 38.81378, 40.17010, 41.50134, 42.93970, 44.37317, 45.79295,
    47.09876, 48.41413, 49.72995, 51.09672, 52.18662, 53.34497, 54.42442, 55.53308,
    56.64611, 57.69032, 58.70046, 59.64953, 60.64692, 61.64533, 62.65312, 63.58665,
    64.59393, 65.55237, 66.57866, 67.46057, 68.31111, 69.16931, 69.88128, 70.59772,
    71.23542, 71.90910, 72.62932, 73.31173, 73.93996, 74.66109, 75.36854, 76.13904,
    76.86075, 77.65669, 78.39602, 79.24672, 80.05577, 80.87842, 81.78743, 82.69322,
    83.54597, 84.43093, 85.29433, 86.11023, 87.01602, 87.96816, 89.07732, 90.21744,
    91.43488, 92.85162, 94.39351, 96.08081, 97.78255, 99.31664, 100.14130, 100.50560,
    100.54830, 100.20930, 99.71041, 99.48242
    ]

    time_dataB5 = [
    0.000, 0.03333, 0.06667, 0.100, 0.133, 0.167, 0.200, 0.233, 0.267, 0.300,
    0.333, 0.367, 0.400, 0.433, 0.467, 0.500, 0.533, 0.567, 0.600, 0.633,
    0.667, 0.700, 0.733, 0.767, 0.800, 0.833, 0.867, 0.900, 0.933, 0.967,
    1.000, 1.033, 1.067, 1.100, 1.133, 1.167, 1.200, 1.233, 1.267, 1.300,
    1.333, 1.367, 1.400, 1.433, 1.467, 1.500, 1.533, 1.567, 1.600, 1.633,
    1.667, 1.700, 1.733, 1.767, 1.800, 1.833, 1.867, 1.900, 1.933, 1.967,
    2.000, 2.033, 2.067, 2.100, 2.133, 2.167, 2.200, 2.233, 2.267, 2.300,
    2.333, 2.367, 2.400, 2.433, 2.467, 2.500, 2.533, 2.567, 2.600, 2.633,
    2.667, 2.700, 2.733, 2.767, 2.800, 2.833, 2.867, 2.900, 2.933, 2.967,
    3.000, 3.033, 3.067, 3.100, 3.133, 3.167, 3.200, 3.233, 3.267, 3.300,
    3.333, 3.367, 3.400, 3.433, 3.467, 3.500, 3.533, 3.567, 3.600, 3.633,
    3.667, 3.700, 3.733, 3.767, 3.800, 3.833, 3.867, 3.900, 3.933, 3.967,
    4.000, 4.033, 4.067, 4.100, 4.133, 4.167, 4.200, 4.233, 4.267, 4.300,
    4.333, 4.367, 4.400, 4.433, 4.467, 4.500, 4.533, 4.567, 4.600, 4.633,
    4.667, 4.700, 4.733, 4.767, 4.800, 4.833, 4.867, 4.900, 4.933, 4.967,
    5.000, 5.033, 5.067
    ]

    distance_dataB5 = [
        54.3, 54.2, 54.2, 54.1, 53.7, 53.2, 52.5, 51.7, 50.9, 50.1,
        49.2, 48.4, 47.4, 46.5, 45.6, 44.6, 43.6, 42.6, 41.8, 40.8,
        39.9, 38.9, 38.1, 37.1, 36.2, 35.4, 34.6, 34.0, 33.2, 32.6,
        32.0, 31.2, 29.7, 29.8, 29.3, 29.4, 29.2, 29.0, 29.0, 29.0,
        29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.2,
        29.4, 29.4, 29.6, 29.5, 29.7, 30.3, 31.2, 31.6, 32.0, 32.4,
        32.8, 33.0, 33.4, 33.6, 34.0, 34.2, 34.4, 34.6, 34.8, 34.9,
        35.1, 35.3, 35.6, 35.7, 35.9, 36.1, 36.4, 36.5, 36.8, 36.9,
        37.1, 37.3, 37.5, 37.6, 37.7, 37.9, 38.1, 38.3, 38.5, 38.7,
        38.9, 39.1, 39.3, 39.5, 39.6, 39.7, 39.9, 40.1, 40.3, 40.5,
        40.7, 40.9, 40.9, 41.1, 41.3, 41.5, 41.6, 41.8, 42.0, 42.2,
        42.4, 42.6, 42.8, 43.0, 43.2, 43.4, 43.6, 43.8, 44.0, 44.2,
        44.4, 44.6, 44.8, 45.1, 45.3, 45.6, 45.8, 46.1, 46.4, 46.8,
        47.2, 47.6, 48.0, 48.5, 49.0, 49.5, 50.1, 50.7, 51.4, 52.1,
        52.7, 53.5, 54.1, 54.3, 54.5, 54.4, 54.3, 54.2, 54.2, 54.2,
        54.2, 54.2, 54.2
    ]

    time_dataD3 = [
        1.433333, 1.466667, 1.500000, 1.533333, 1.566667, 1.600000, 1.633333, 1.666667,
        1.700000, 1.733333, 1.766667, 1.800000, 1.833333, 1.866667, 1.900000, 1.933333,
        1.966667, 2.000000, 2.033333, 2.066667, 2.100000, 2.133333, 2.166667, 2.200000,
        2.233333, 2.266667, 2.300000, 2.333333, 2.366667, 2.400000, 2.433333, 2.466667,
        2.500000, 2.533333, 2.566667, 2.600000, 2.633333, 2.666667, 2.700000, 2.733333,
        2.766667, 2.800000, 2.833333, 2.866667, 2.900000, 2.933333, 2.966667, 3.000000,
        3.033333, 3.066667, 3.100000, 3.133333, 3.166667, 3.200000, 3.233333, 3.266667,
        3.300000, 3.333333, 3.366667, 3.400000, 3.433333, 3.466667, 3.500000, 3.533333,
        3.566667, 3.600000, 3.633333, 3.666667, 3.700000, 3.733333, 3.766667, 3.800000,
        3.833333, 3.900000, 3.933333, 4.000000, 4.033333, 4.066667, 4.100000, 4.133333,
        4.166667, 4.200000, 4.233333, 4.266667, 4.300000, 4.333333, 4.366667, 4.400000,
        4.433333
    ]



    distance_dataD3 = [
        55.47127, 55.72363, 55.87127, 55.97207, 56.15722, 56.28918, 56.49608, 56.69378,
        56.79403, 56.95904, 57.10612, 57.29765, 57.39600, 57.59598, 57.79267, 58.05564,
        58.34827, 58.61041, 59.02482, 59.48524, 59.94237, 60.48288, 61.10867, 61.80809,
        62.47563, 63.24230, 64.12682, 65.05456, 66.09949, 67.08247, 68.24963, 69.04654,
        70.14189, 71.41792, 72.51147, 73.79211, 75.06223, 76.31363, 77.64325, 79.07369,
        80.41826, 81.96715, 83.33277, 84.70190, 86.06679, 87.59418, 88.94667, 90.05253,
        90.99310, 92.02723, 92.86307, 93.36580, 93.78471, 94.19338, 94.52490, 94.75691,
        95.01865, 95.29733, 95.50949, 95.78316, 96.14980, 96.01023, 96.88193, 96.72118,
        97.95751, 98.23941, 98.49621, 98.66279, 98.80547, 98.94128, 99.04144, 99.11627,
        99.19248, 99.20637, 99.00202, 99.31536, 99.39498, 99.45813, 99.62829, 98.97698,
        100.99550, 100.45520, 100.29470, 100.29500, 100.21890, 99.86885, 99.23995, 97.99184, 97.99184
    ]

    time_dataD4 = [
        32.56667, 32.60000, 32.63333, 32.66667, 32.70000, 32.73333, 32.76667, 32.80000,
        32.83333, 32.86667, 32.90000, 32.93333, 32.96667, 33.00000, 33.03333, 33.06667,
        33.10000, 33.13333, 33.16667, 33.20000, 33.23333, 33.26667, 33.30000, 33.33333,
        33.36667, 33.40000, 33.43333, 33.46667, 33.50000, 33.53333, 33.56667, 33.60000,
        33.63333, 33.66667, 33.70000, 33.73333, 33.76667, 33.80000, 33.83333, 33.86667,
        33.90000, 33.93333, 33.96667, 34.00000, 34.03333, 34.06667, 34.10000, 34.13333,
        34.16667, 34.20000, 34.23333, 34.26667, 34.30000, 34.33333, 34.36667, 34.40000,
        34.43333, 34.46667, 34.50000, 34.53333, 34.56667, 34.60000, 34.63333, 34.66667,
        34.70000, 34.73333, 34.76667, 34.80000, 34.83333, 34.86667, 34.90000, 34.93333,
        34.96667, 35.00000, 35.03333, 35.06667, 35.10000, 35.13333, 35.16667, 35.20000,
        35.23333, 35.26667, 35.30000, 35.33333, 35.36667, 35.40000, 35.43333, 35.46667,
        35.50000, 35.53333, 35.56667, 35.60000, 35.63333, 35.66667, 35.70000, 35.73333,
        35.76667, 35.80000, 35.83333, 35.86667, 35.90000, 35.93333, 35.96667, 36.00000
    ]

    distance_dataD4 = [
        54.69487, 54.77368, 54.74549, 54.78717, 54.82117, 54.77329, 54.84831, 54.99828,
        55.03895, 55.14323, 55.14421, 55.34602, 55.46808, 55.50659, 55.80978, 55.84957,
        56.23027, 56.48137, 56.86313, 57.22825, 57.83644, 57.97789, 58.00011, 58.07890,
        58.15657, 58.26422, 58.43667, 58.53828, 58.61084, 58.70838, 58.82079, 59.05616,
        59.22084, 59.50895, 59.74966, 59.85370, 59.96359, 60.12338, 60.28859, 60.37462,
        60.43692, 60.46362, 60.59307, 60.78167, 60.97854, 61.10512, 61.19449, 61.27291,
        61.43937, 61.58855, 61.58642, 61.74232, 61.80416, 61.91716, 62.02709, 62.17424,
        62.21835, 62.34878, 62.53151, 62.59848, 62.64273, 62.72076, 62.81885, 62.91303,
        62.94469, 63.01938, 63.05350, 63.09573, 63.14455, 63.21416, 63.28366, 63.41924,
        63.52613, 63.58623, 63.68467, 63.78468, 63.86078, 63.99977, 64.01235, 64.09558,
        64.09077, 64.14479, 64.23250, 64.27712, 64.33016, 64.44738, 64.48655, 64.58570
    ]


    print(len(time_dataD4), len(distance_dataD4))
    analyze_lift(time_dataD3, distance_dataD3)


#MARK: - Main
def main():
    """Main program execution."""
    mqtt_handler = MQTTHandler() #FIXME: Make it so that it can take more data in once the button is in the pressed state
    time_data = []
    distance_data = []
    
    # Get RPE threshold from user
    rpe_threshold = float(input("Enter RPE threshold (1-10): "))
    while not 1 <= rpe_threshold <= 10:
        rpe_threshold = float(input("Invalid. Enter RPE threshold (1-10): "))
    
    try:
        mqtt_handler.start()
        print("Assume button has been pressed ...")
        
        collecting = True
        print("Data collection started...")
        while collecting:
            # if mqtt_handler.button_pressed:  # Add button state to MQTTHandler
            #     collecting = True
            #     mqtt_handler.data_buffer = {'time': [], 'distance': []}
            if collecting and len(mqtt_handler.data_buffer['time']) > 0:
                min_length = min(len(mqtt_handler.data_buffer['time']), 
                               len(mqtt_handler.data_buffer['distance']))          
                time_data = mqtt_handler.data_buffer['time'][:min_length]
                distance_data = mqtt_handler.data_buffer['distance'][:min_length]

                #FIXME: Remove length argument
                if check_acceleration_stop(distance_data) or len(time_data) > 200: #Data will continue showing on the terminal, but it won't be added to time_data or distance_data
                    collecting = False
                    print("Data collection stopped.")
                    mqtt_handler.data_buffer = {'time': [], 'distance': []}
                    mqtt_handler.stop()
                    
            time.sleep(0.1)
        print("Analyzing data...")
        # Filter the data so that we only analyze times after the button is pressed 
        valid_indices = [i for i, t in enumerate(mqtt_data.time_data) if t >= start_time]
    
        # Filter both lists using the valid indices (if this doesn't work get rid of the next two lines and put time and distance as inputs for analysis)
        filtered_time = [mqtt_data.time_data[i] for i in valid_indices]
        filtered_distance = [mqtt_data.distance_data[i] for i in valid_indices]

        if len(time_data) > 5:  # Only analyze when we have enough data
            analyze_lift(filtered_time, filtered_distance, rpe_threshold)
            save_data(filtered_time, filtered_distance)
        else:
            print("\nNot enough data to analyze. Please try again.\n")
        mqtt_handler.stop()
    except KeyboardInterrupt:
        print("\nStopping Program...")
        mqtt_handler.stop()
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()

#run mosquitto with /usr/local/opt/mosquitto/sbin/mosquitto -c /usr/local/etc/mosquitto/etc/mosquitto/mosquitto.conf
