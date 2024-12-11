#include <Servo.h>
#include <WiFiNINA.h>
#include <ArduinoMqttClient.h>

// Reconnect parameters
const int maxAttempts = 5;
int reconnectAttempts = 0;
unsigned long lastReconnectAttempt = 0; 
const long reconnectInterval = 1000;

// WiFi credentials
const char ssid[] = "BU Guest (unencrypted)";

// MQTT settings
const char broker[] = "10.193.180.134";
const int port = 1883;
const char topic[] = "lift_data";
const char servo_topic[] = "servoMotor";

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

const int sensorPin = A0;
const int servoPin = 9;
const int buttonPin = 2; 
int lastButtonState = LOW;


Servo myservo;
float distanceCM;
unsigned long startTime;

void setup() {
  Serial.begin(9600);
  
  myservo.attach(servoPin);
  myservo.write(0);
  
  Serial.print("Connecting to WiFi");
  while (WiFi.begin(ssid) != WL_CONNECTED) {
    Serial.print(".");
    delay(5000);
  }
  Serial.println("\nConnected to WiFi");

  if (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());
    return;
  }
  
  Serial.println("Connected to MQTT broker");
  mqttClient.onMessage(onMqttMessage);
  mqttClient.subscribe("servoMotor");
  startTime = millis();
}

void loop() {
  mqttClient.poll();

  // Read sensor and send data continuously
  int sensorValue = analogRead(sensorPin);
  distanceCM = analogToDistance(sensorValue);
  // Serial.print("Time: ");
  // Serial.print(String((millis() - startTime)) );
  // Serial.print(" s, Distance: ");
  // Serial.print(distanceCM);
  // Serial.println(" cm");
  
  // Send data via MQTT
  String message = "{\"time\":" + String((millis() - startTime)) 
                + ",\"distance\":" + String(distanceCM) + "}";
  
  mqttClient.beginMessage(topic);
  mqttClient.print(message);
  mqttClient.endMessage();

  int buttonState = digitalRead(buttonPin);
  // Debounce logic is a little flawed, will be fixed in a different lifetime
  if (buttonState != lastButtonState) {
    if (buttonState == HIGH) {
      Serial.println("Button pressed");
      // Send MQTT message on button press

      String jsonMessage = "{\"start_time\":" + String((millis() - startTime)/1000.0) + "}";
      mqttClient.beginMessage("start_time");
      mqttClient.print(jsonMessage);
      mqttClient.endMessage();
    }
    delay(100);  
  }
  lastButtonState = buttonState;
  // Handle incoming servo commands
  int messageSize = mqttClient.parseMessage();

  // Handle MQTT reconnection
  if (!mqttClient.connected()) {
    unsigned long now = millis();
    if (now - lastReconnectAttempt > reconnectInterval) {
      lastReconnectAttempt = now;
      if (mqttClient.connect(broker, port)) {
        mqttClient.subscribe(servo_topic);
        Serial.println("Reconnected to MQTT broker");
      }
    }
  }
  

  delay(100);
}
// Function to convert analog value to distance (in cm)
float analogToDistance(int analogValue) {
  const float analogReferenceVoltage = 5.0;
  float voltage = analogValue * (analogReferenceVoltage / 1023.0);
  float distance;
  
  // Segment 1: 0 < x < 10 cm
  if (voltage <= 2.3) {
    distance = (10.0 / 2.3) * voltage;
  }
  // Segment 2: 10 < x < 15 cm
  else if (voltage > 2.3 && voltage <= 2.3 + 0.09 * (15 - 10)) {
    distance = ((voltage - 2.3) / 0.09) + 10.0;
  }
  // Segment 3: x > 15 cm
  else {
    distance = pow(30.07109709 / voltage, 1.0 / 0.8265);
  }
  
  return distance;
}

void onMqttMessage(int messageSize) {
  if (messageSize) {
    String incomingTopic = mqttClient.messageTopic();
    String payload = mqttClient.readString();
    
    // Trim whitespace and newlines
    payload.trim();
    
    Serial.println("Debug MQTT Message:");
    Serial.print("Topic: '");
    Serial.print(incomingTopic);
    Serial.println("'");
    Serial.print("Payload: '");
    Serial.print(payload);
    Serial.println("'");
    
    // Use equals() for string comparison
    if (incomingTopic.equals("servoMotor")) {
        if (payload.equals("HIGH")) {
            Serial.println("Matched HIGH - Setting Servo to 180");
            myservo.write(180);
        } 
        else if (payload.equals("LOW")) {
            Serial.println("Matched LOW - Setting Servo to 0");
            myservo.write(0);
        }
        else {
            Serial.print("Unrecognized payload: '");
            Serial.print(payload);
            Serial.println("'");
            // Print ASCII values for debugging
            for (int i = 0; i < payload.length(); i++) {
                Serial.print((int)payload[i]);
                Serial.print(" ");
            }
            Serial.println();
          }
        }
      }
}
