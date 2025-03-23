#include <ESP32Servo.h>

Servo servo;  
const int servoPin = 26;  // Change this to your actual servo GPIO pin
const int default_angle = 75;  

void setup()
{
    Serial.begin(115200);
    servo.attach(servoPin);
    servo.write(default_angle);  // Set initial position
}

byte angle;
byte pre_angle = default_angle;
long t = millis();

void loop()
{
    if (Serial.available() > 0)
    {
        Serial.readBytes(&angle, 1);  // Read a single byte
        if (angle != pre_angle)
        {
            servo.write(angle);
            pre_angle = angle;
        }
        t = millis();
    }

    if (millis() - t > 1000)  // Reset position if no new command in 1 sec
    {
        servo.write(default_angle);
        pre_angle = default_angle;
    }
}
