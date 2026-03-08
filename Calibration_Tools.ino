/**
 * ItsyBitsy M4 - Calibration Tool
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define PCA9685_ADDR 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

const int CHANNELS[6] = {0, 1, 2, 3, 4, 5};
const char* SERVO_NAMES[6] = {"BASE", "SHOULDER", "ELBOW", "WRIST", "GRIP_ROT", "GRIPPER"};

uint16_t servoMin[6] = {150, 150, 150, 150, 150, 150};
uint16_t servoMax[6] = {600, 600, 600, 600, 600, 600};
int currentAngles[6] = {90, 90, 90, 90, 90, 90};

#define SERVO_FREQ 50

void setServoAngle(int servoIdx, int angle) {
  angle = constrain(angle, 0, 180);
  uint16_t pulse = map(angle, 0, 180, servoMin[servoIdx], servoMax[servoIdx]);
  pwm.setPWM(CHANNELS[servoIdx], 0, pulse);
  currentAngles[servoIdx] = angle;
}

void moveAll(int angles[6]) {
  for (int i = 0; i < 6; i++) setServoAngle(i, angles[i]);
}

void sweepServo(int servoIdx, int fromAngle, int toAngle, int delayMs) {
  int step = (toAngle > fromAngle) ? 1 : -1;
  for (int a = fromAngle; a != toAngle; a += step) {
    setServoAngle(servoIdx, a);
    delay(delayMs);
  }
  setServoAngle(servoIdx, toAngle);
}

void printCurrentAngles() {
  Serial.println("CURRENT ANGLES:");
  for (int i = 0; i < 6; i++) {
    Serial.print("  Servo "); Serial.print(i);
    Serial.print(" ("); Serial.print(SERVO_NAMES[i]); Serial.print(")");
    Serial.print(" -> "); Serial.print(currentAngles[i]); Serial.println(" deg");
  }
}

void printCalibrationValues() {
  Serial.println("CURRENT PWM CALIBRATION:");
  Serial.print("  servoMin[6] = {");
  for (int i = 0; i < 6; i++) { Serial.print(servoMin[i]); if (i < 5) Serial.print(", "); }
  Serial.println("};");
  Serial.print("  servoMax[6] = {");
  for (int i = 0; i < 6; i++) { Serial.print(servoMax[i]); if (i < 5) Serial.print(", "); }
  Serial.println("};");
}

void printHelp() {
  Serial.println("CALIBRATION TOOL - COMMAND MENU");
  Serial.println("MOVE [servo] [angle]         Move one servo");
  Serial.println("  Example: MOVE 0 90");
  Serial.println("  Servos: 0=BASE 1=SHOULDER 2=ELBOW");
  Serial.println("          3=WRIST 4=GRIP_ROT 5=GRIPPER");
  Serial.println("ALL [a] [b] [c] [d] [e] [f]  Move all 6 servos");
  Serial.println("  Example: ALL 90 45 60 30 90 90");
  Serial.println("PRINT                        Show current angles");
  Serial.println("CENTER                       Move all to 90 degrees");
  Serial.println("SWEEP [servo]                Sweep servo 0->180->0");
  Serial.println("  Example: SWEEP 1");
  Serial.println("MIN [servo] [value]          Set min PWM value");
  Serial.println("MAX [servo] [value]          Set max PWM value");
  Serial.println("PRINTCAL                     Show PWM calibration values");
  Serial.println("SAVEHOME                     Save current as HOME");
  Serial.println("SAVEBIN [name]               Save current as bin position");
  Serial.println("  Example: SAVEBIN PAPER");
  Serial.println("SAVEIK [cx] [cy] [w]         Save current as IK table entry");
  Serial.println("  Example: SAVEIK 320 240 80");
}

void saveHome() {
  Serial.println("-- COPY THIS INTO YOUR MAIN SKETCH --");
  Serial.print("const int HOME_ANGLES[6] = {");
  for (int i = 0; i < 6; i++) { Serial.print(currentAngles[i]); if (i < 5) Serial.print(", "); }
  Serial.println("};");
}

void saveBin(const String& binName) {
  Serial.println("-- COPY THIS INTO YOUR MAIN SKETCH --");
  Serial.print("BinPosition BIN_"); Serial.print(binName);
  Serial.print(" = {\""); Serial.print(binName); Serial.print("\", {");
  for (int i = 0; i < 6; i++) { Serial.print(currentAngles[i]); if (i < 5) Serial.print(", "); }
  Serial.println("}};");
}

void saveIK(int cx, int cy, int w) {
  Serial.println("-- ADD THIS LINE TO IK_TABLE --");
  Serial.print("  {");
  Serial.print(cx); Serial.print(", ");
  Serial.print(cy); Serial.print(", ");
  Serial.print(w); Serial.print(",  {");
  for (int i = 0; i < 6; i++) { Serial.print(currentAngles[i]); if (i < 5) Serial.print(", "); }
  Serial.println("}},");
}

void processCommand(String raw) {
  raw.trim();
  if (raw.length() == 0) return;

  int sp = raw.indexOf(' ');
  String cmd = (sp > 0) ? raw.substring(0, sp) : raw;
  cmd.toUpperCase();
  String args = (sp > 0) ? raw.substring(sp + 1) : "";
  args.trim();

  if (cmd == "MOVE") {
    int servo, angle;
    if (sscanf(args.c_str(), "%d %d", &servo, &angle) == 2) {
      if (servo < 0 || servo > 5) { Serial.println("ERROR: Servo must be 0-5"); return; }
      setServoAngle(servo, angle);
      Serial.print("Moved servo "); Serial.print(servo);
      Serial.print(" ("); Serial.print(SERVO_NAMES[servo]); Serial.print(")");
      Serial.print(" to "); Serial.print(angle); Serial.println(" deg");
    } else { Serial.println("ERROR: Usage: MOVE [servo 0-5] [angle 0-180]"); }
  }
  else if (cmd == "ALL") {
    int a[6];
    if (sscanf(args.c_str(), "%d %d %d %d %d %d", &a[0],&a[1],&a[2],&a[3],&a[4],&a[5]) == 6) {
      moveAll(a); Serial.println("Moved all servos."); printCurrentAngles();
    } else { Serial.println("ERROR: Usage: ALL [a0] [a1] [a2] [a3] [a4] [a5]"); }
  }
  else if (cmd == "PRINT")   { printCurrentAngles(); }
  else if (cmd == "CENTER")  { int c[6]={90,90,90,90,90,90}; moveAll(c); Serial.println("All servos at 90 deg."); }
  else if (cmd == "SWEEP") {
    int servo = args.toInt();
    if (servo < 0 || servo > 5) { Serial.println("ERROR: Servo must be 0-5"); return; }
    Serial.print("Sweeping servo "); Serial.print(servo); Serial.print(" ("); Serial.print(SERVO_NAMES[servo]); Serial.println(")...");
    sweepServo(servo, 0, 180, 15); delay(500);
    sweepServo(servo, 180, 0, 15); delay(500);
    setServoAngle(servo, 90);
    Serial.println("Sweep done."); printCalibrationValues();
  }
  else if (cmd == "MIN") {
    int servo, val;
    if (sscanf(args.c_str(), "%d %d", &servo, &val) == 2) {
      if (servo < 0 || servo > 5) { Serial.println("ERROR: Servo must be 0-5"); return; }
      servoMin[servo] = val;
      Serial.print("Set servoMin["); Serial.print(servo); Serial.print("] = "); Serial.println(val);
      setServoAngle(servo, currentAngles[servo]); printCalibrationValues();
    } else { Serial.println("ERROR: Usage: MIN [servo 0-5] [pwm value]"); }
  }
  else if (cmd == "MAX") {
    int servo, val;
    if (sscanf(args.c_str(), "%d %d", &servo, &val) == 2) {
      if (servo < 0 || servo > 5) { Serial.println("ERROR: Servo must be 0-5"); return; }
      servoMax[servo] = val;
      Serial.print("Set servoMax["); Serial.print(servo); Serial.print("] = "); Serial.println(val);
      setServoAngle(servo, currentAngles[servo]); printCalibrationValues();
    } else { Serial.println("ERROR: Usage: MAX [servo 0-5] [pwm value]"); }
  }
  else if (cmd == "PRINTCAL") { printCalibrationValues(); }
  else if (cmd == "SAVEHOME") { saveHome(); }
  else if (cmd == "SAVEBIN") {
    if (args.length() == 0) { Serial.println("ERROR: Usage: SAVEBIN [PAPER|PLASTIC|GENERAL]"); return; }
    args.toUpperCase(); saveBin(args);
  }
  else if (cmd == "SAVEIK") {
    int cx, cy, w;
    if (sscanf(args.c_str(), "%d %d %d", &cx, &cy, &w) == 3) { saveIK(cx, cy, w); }
    else { Serial.println("ERROR: Usage: SAVEIK [cx] [cy] [bbox_width]"); }
  }
  else if (cmd == "HELP") { printHelp(); }
  else { Serial.print("Unknown command: "); Serial.println(cmd); Serial.println("Type HELP to see all commands."); }
}

void setup() {
  Serial.begin(9600);
  while (!Serial) delay(10);
  Wire.begin();
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  delay(10);
  int center[6] = {90, 90, 90, 90, 90, 90};
  moveAll(center);
  Serial.println();
  printHelp();
  Serial.println();
  Serial.println("All servos at 90 deg. Ready for commands.");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    processCommand(line);
  }
}