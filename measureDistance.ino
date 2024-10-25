const int sensorPin = A0;  // Analog pin A0 connected to the sensor
float distanceCM;          
unsigned long startTime;   

void setup() {
  Serial.begin(9600);

  pinMode(sensorPin, INPUT);
  
  //get start time
  startTime = millis();
  
  Serial.println("IR Distance Sensor Monitoring Started...");
  Serial.println("Time (s)\tDistance (cm)");  // Header for the output
}

void loop() {
  int sensorValue = analogRead(sensorPin);
  
  //convert analog value to voltage
  float voltage = sensorValue * (5.0 / 1023.0);
  
  //calculate the distance based on the sensor value
  distanceCM = analogToDistance(sensorValue);
  
  //calculate the elapsed time since the program started
  float elapsedTime = (millis() - startTime) / 1000.0;  //convert to seconds
  
  //display the time, voltage, and distance
  Serial.print(elapsedTime, 2);  
  Serial.print("\t");
  Serial.print(voltage, 2);  
  Serial.print("\t");
  Serial.print(distanceCM);
  Serial.println(" cm");
  
  delay(100); 
}

//function to convert analog value to distance (in cm)
float analogToDistance(int analogValue) {
  float voltage = analogValue * (5.0 / 1023.0);
  float k = 27;  // scaling constant
  float b = 1.2;   

  float distance = pow(k*(1/voltage),b);
  return distance;
}
