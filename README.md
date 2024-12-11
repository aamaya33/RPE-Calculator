# RPE-Calculator

This project integrates IoT devices and software to collect, process, and analyze lift data using an infrared distance sensor and MQTT protocol. The system measures the distance of a moving object over time and provides analysis to evaluate performance.

## Project Components

### 1. `main.py`
- A Python script to collect and analyze lift data in real-time.
- Connects to an MQTT broker to receive distance data.
- Features:
  - **Real-Time Analysis**: Determines stopping points and acceleration.
  - **Custom Thresholds**: Accepts user-defined thresholds for analysis.
  - **Data Visualization**: Supports graphing lift metrics (velocity v time and position v time) using `matplotlib`.

### 2. `measureDistance.ino`
- An Arduino sketch for interfacing with an ultrasonic distance sensor and servo motor.
- Publishes sensor readings to an MQTT topic (`lift_data`) at regular intervals.
- Features:
  - **WiFi Connectivity**: Uses the WiFiNINA library to connect to a network.
  - **Servo Motor Control**: Responds to MQTT messages to adjust the servo angle.
  - **Button Press Detection**: Publishes a start signal upon button press.

### 3. `MQTT.py`
- A Python library for handling MQTT connections and data processing.
- Features:
  - **Data Buffering**: Stores received time and distance data.
  - **Error Handling**: Handles connection failures and reconnections.
  - **Topic Subscription**: Listens for lift data and start-time signals.

## How to Use

### Setup
1. **Hardware**:
   - Arduino Nano 33 board with a compatible infrared sensor and servo motor.
   - Wi-Fi network and MQTT broker (used mosquitto).
2. **Software**:
   - Install dependencies:
     paho-mqtt
     matplotlib
     numpy
     scipy
   - Flash `measureDistance.ino` onto the Arduino.

### Run the Project
1. Start the MQTT broker on your local machine or a dedicated server. Make sure that the config file (/etc/mosquitto/mosquitto.conf) has the following:
   - listener 1883
   - allow_anonymous true

3. Launch the `main.py` script and turn on arduino to begin data collection and analysis.
