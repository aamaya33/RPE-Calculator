import paho.mqtt.client as mqtt
import json
import time
from typing import Optional, Dict, List

class MQTTHandler:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.data_buffer: Dict[str, List[float]] = {'time': [], 'distance': []}
        self.connected = False
        # self.button_pressed = True
        self.last_message_time = time.time()
        self.disconnect_timeout = 2.0
        
    def on_connect(self, client, userdata, flags, rc):
        self.connected = True
        client.subscribe("lift_data")
    
    def check_connection_status(self):
        if self.connected and (time.time() - self.last_message_time) > self.disconnect_timeout:
            return True
        return False

    def publish(self, topic: str, message: str):
        """Publish a message to specified MQTT topic"""
        if not self.connected:
            print("Warning: No longer connected to MQTT broker")
            return False
            
        try:
            result = self.client.publish(topic, message)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Published {message} to {topic}")
                return True
            else:
                print(f"Failed to publish message: {result.rc}")
                return False
        except Exception as e:
            print(f"Error publishing message: {e}")
            return False
        
    def on_message(self, client, userdata, msg):
        self.last_message_time = time.time()
        try:
            data = json.loads(msg.payload.decode())
            
            if msg.topic == "start_time":
                self.start_time = float(data['start_time'])
                print(f"Received start time: {self.start_time}")
                return
                
            if 'status' in data:
                return
            
            self.data_buffer['time'].append(float(data['time']) / 1000.0)
            self.data_buffer['distance'].append(float(data['distance']))
            # if (not connected): 
            print(f"Time: {self.data_buffer['time'][-1]:.2f}s, Distance: {self.data_buffer['distance'][-1]}cm")
        except json.JSONDecodeError as e:
            print(f"Error parsing MQTT message: {e}")
            
    def start(self, broker="localhost", port=1883, retries=10):
        for attempt in range(retries):
            try:
                print(f"Attempting to connect to MQTT broker at {broker}:{port} (Attempt {attempt + 1}/{retries})")
                self.client.connect(broker, port, 60)
                self.client.loop_start()
                print("Successfully connected to MQTT broker")
                return
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("Max retries reached. Could not connect to MQTT broker")
                    raise
        self.connected = True 
            
    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.disconnected = True
        print("Arduino disconnected. Analyzing final data...")
