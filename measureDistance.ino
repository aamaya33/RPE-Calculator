//set up logic for the IR sensor
const int sensorPin = A0;  
float distanceCM;
unsigned long startTime;

//set up logic for the button
const int buttonPin = 2;
bool buttonPressed = false; 
bool lastButtonState = false; //previous button state for debouncing

float distanceSum = 0; 
int distanceCount = 0; 

unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50; //ms 

void setup() {
  Serial.begin(9600);
  pinMode(buttonPin, INPUT_PULLUP); //set up button as input. IR sensor doens't have to be an input since serial ports are already inputs. 

  startTime = millis();

  Serial.println("IR Distance Sensor Monitoring Started...");
  Serial.println("Time (s)\tVoltage (V)\tDistance (cm)"); 
}

void loop() {
  int sensorValue = analogRead(sensorPin);

  //convert analog value to voltage
  const float analogReferenceVoltage = 5.0;
  float voltage = sensorValue * (analogReferenceVoltage / 1023.0);

  //calculate the distance based on the sensor value
  distanceCM = analogToDistance(sensorValue);

  //read the current button state
  bool reading = !digitalRead(buttonPin); //true when pressed

  //debounce logic
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonPressed) {
      buttonPressed = reading;

      if (buttonPressed) {
        //button was just pressed
        distanceSum = 0;
        distanceCount = 0;
      } else {
        //button was just released
        setCoords(); 
      }
    }
  }

  lastButtonState = reading;

  if (buttonPressed) {
    distanceSum += distanceCM;
    distanceCount++;
  }

  //calculate the elapsed time since the program started
  float elapsedTime = (millis() - startTime) / 1000.0;  // Convert to seconds

  Serial.print(elapsedTime, 2);
  Serial.print("\t");
  Serial.print(voltage, 2);
  Serial.print("\t");
  Serial.print(distanceCM);
  Serial.println(" cm");

  delay(100);
}

// Function to convert analog value to distance (in cm)
float analogToDistance(int analogValue) {
  const float analogReferenceVoltage = 5.0;
  float voltage = analogValue * (analogReferenceVoltage / 1023.0);
  float k = 27;  // Scaling constant
  float b = 1.2;

  if (voltage < 0.1) {
    return 120.0; //return max distance if out of range. 
  }

  float distance = pow(k / voltage, b);
  return distance;
}

void setCoords() {
  if (distanceCount == 0) {
    Serial.println("No data collected.");
    return;
  }

  float y1 = distanceSum / distanceCount;
  Serial.print("Y1 is calculated to be: ");
  Serial.println(y1);

  distanceSum = 0;
  distanceCount = 0;
}
