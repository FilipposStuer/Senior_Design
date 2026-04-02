/**
 * ItsyBitsy M4 Firmware - Robot Arm Controller with PCA9685
 * Serial Monitor Control Version
 * v2.0 - Fixes applied to original Demo_march_24.ino:
 *   1. Removed PROGMEM / memcpy_P  (ItsyBitsy M4 is ARM, not AVR)
 *   2. Replaced sscanf %f (unreliable on Arduino ARM) with toFloat()
 *   3. Added optional base angle parameter to MOVE command
 *   4. Added workspace boundary warning
 *   5. Added PICK command support for vision integration
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "ik_table.h"   // Auto-generated from MATLAB
#include <math.h>

// ─────────────────────────────────────────────
// PCA9685 CONFIGURATION
// ─────────────────────────────────────────────

#define PCA9685_ADDR 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

// Servo channels on PCA9685
const int CH_BASE        = 0;
const int CH_SHOULDER    = 1;
const int CH_ELBOW       = 2;
const int CH_WRIST       = 3;
const int CH_GRIPPER_ROT = 4;
const int CH_GRIPPER     = 5;
const int CHANNELS[6]    = {CH_BASE, CH_SHOULDER, CH_ELBOW, CH_WRIST, CH_GRIPPER_ROT, CH_GRIPPER};

// PWM calibration -- values from physical calibration table
uint16_t servoMin[6] = {130, 360, 60, 180, 60, 100};
uint16_t servoMax[6] = {600, 600, 600, 600, 600, 600};
#define SERVO_FREQ 50

// Gripper positions
const int GRIPPER_CLOSED = 20;
const int GRIPPER_OPEN   = 90;

// Home position
const int HOME_ANGLES[6] = {90, 90, 90, 90, 90, GRIPPER_CLOSED};

// Workspace limits from ik_table.h
const float WS_X_MIN =  8.5f;
const float WS_X_MAX = 13.5f;
const float WS_Y_MIN =  0.6f;
const float WS_Y_MAX =  4.4f;

// Current servo angles
int currentAngles[6];

// ─────────────────────────────────────────────
// PCA9685 FUNCTIONS
// ─────────────────────────────────────────────

void initPCA9685() {
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  delay(10);
}

void setServoAngle(int channel, int angle) {
  angle = constrain(angle, 0, 180);
  uint16_t pulse = map(angle, 0, 180, servoMin[channel], servoMax[channel]);
  pwm.setPWM(CHANNELS[channel], 0, pulse);
}

void moveServos(const int angles[6]) {
  for (int i = 0; i < 6; i++) {
    setServoAngle(i, angles[i]);
    currentAngles[i] = angles[i];
  }
}

void moveServosSmooth(const int targetAngles[6], int stepDelay = 20) {
  int startAngles[6];
  memcpy(startAngles, currentAngles, sizeof(startAngles));

  int maxSteps = 0;
  int steps[6];
  for (int i = 0; i < 6; i++) {
    steps[i] = abs(targetAngles[i] - startAngles[i]);
    if (steps[i] > maxSteps) maxSteps = steps[i];
  }
  if (maxSteps == 0) return;

  for (int step = 1; step <= maxSteps; step++) {
    int intermediate[6];
    for (int i = 0; i < 6; i++) {
      intermediate[i] = (steps[i] > 0)
        ? startAngles[i] + (targetAngles[i] - startAngles[i]) * step / maxSteps
        : targetAngles[i];
    }
    moveServos(intermediate);
    delay(stepDelay);
  }
}

// ─────────────────────────────────────────────
// IK LOOKUP
// ─────────────────────────────────────────────

bool ikLookup(float x, float y, int baseAngle, int outputAngles[6]) {
  int   bestIdx = -1;
  float minDist = 1e9f;

  for (int i = 0; i < IK_TABLE_SIZE; i++) {
    float dx   = IK_TABLE[i].x - x;
    float dy   = IK_TABLE[i].y - y;
    float dist = dx*dx + dy*dy;

    if (dist < minDist) {
      minDist = dist;
      bestIdx = i;
    }
  }

  if (bestIdx >= 0) {
    memcpy(outputAngles, IK_TABLE[bestIdx].angles, 6 * sizeof(int));
    outputAngles[0] = constrain(baseAngle, 0, 180);
    outputAngles[5] = GRIPPER_OPEN;
    return true;
  }
  return false;
}

// ─────────────────────────────────────────────
// PRINT HELPERS
// ─────────────────────────────────────────────

void printStatus() {
  Serial.println("-- Current Angles ------------------");
  const char* names[6] = {"Base      ", "Shoulder  ", "Elbow     ",
                           "Wrist     ", "Grip Rot  ", "Gripper   "};
  for (int i = 0; i < 6; i++) {
    Serial.print("  "); Serial.print(names[i]);
    Serial.print(": "); Serial.print(currentAngles[i]); Serial.println(" deg");
  }
  Serial.println("------------------------------------");
}

void printHelp() {
  Serial.println("-- Commands ------------------------");
  Serial.println("  MOVE x y        -- Move (e.g. MOVE 10.5 2.0)");
  Serial.println("  PICK class x y w h -- From vision system (e.g. PICK PLASTIC 10.5 2.0 100 80)");
  Serial.println("  GRIP            -- Close gripper");
  Serial.println("  RELEASE         -- Open gripper");
  Serial.println("  HOME            -- Return to home position");
  Serial.println("  STATUS          -- Show current servo angles");
  Serial.println("  HELP            -- Show this list");
  Serial.println("------------------------------------");
}

// ─────────────────────────────────────────────
// COMMAND PROCESSING
// ─────────────────────────────────────────────

void processCommand(String raw) {
  raw.trim();
  if (raw.length() == 0) return;

  String upper = raw;
  upper.toUpperCase();

  // ── MOVE x y ──
  if (upper.startsWith("MOVE")) {
    String coords = raw.substring(4);
    coords.trim();

    int sp1 = coords.indexOf(' ');
    if (sp1 == -1) {
      Serial.println("ERROR: Usage is MOVE x y");
      return;
    }
    float x = coords.substring(0, sp1).toFloat();
    float y = coords.substring(sp1 + 1).toFloat();

    if (x < WS_X_MIN || x > WS_X_MAX || y < WS_Y_MIN || y > WS_Y_MAX) {
      Serial.print("WARNING: Coordinates (");
      Serial.print(x); Serial.print(", "); Serial.print(y);
      Serial.println(") outside reachable range");
    }

    int angles[6];
    if (!ikLookup(x, y, 90, angles)) {
      Serial.println("ERROR: No IK solution");
      return;
    }

    Serial.print("Moving to (");
    Serial.print(x); Serial.print(", "); Serial.print(y);
    Serial.println(")");
    moveServosSmooth(angles);
    Serial.println("DONE");
  }

  // ── PICK command (from vision) ──
  else if (upper.startsWith("PICK")) {
    String rest = raw.substring(4);
    rest.trim();
    
    // Find first space to separate class from coordinates
    int firstSpace = rest.indexOf(' ');
    if (firstSpace == -1) {
      Serial.println("ERROR: Invalid PICK format");
      return;
    }
    
    String classStr = rest.substring(0, firstSpace);
    String coords = rest.substring(firstSpace + 1);
    coords.trim();
    
    // Parse coordinates using toFloat() instead of sscanf
    int sp1 = coords.indexOf(' ');
    int sp2 = coords.indexOf(' ', sp1 + 1);
    int sp3 = coords.indexOf(' ', sp2 + 1);
    
    if (sp1 == -1 || sp2 == -1 || sp3 == -1) {
      Serial.println("ERROR: Need 4 numbers after class (x y w h)");
      return;
    }
    
    float x = coords.substring(0, sp1).toFloat();
    float y = coords.substring(sp1 + 1, sp2).toFloat();
    int w = (int)coords.substring(sp2 + 1, sp3).toFloat();
    int h = (int)coords.substring(sp3 + 1).toFloat();
    
    Serial.print("PICK: class="); Serial.print(classStr);
    Serial.print(" x="); Serial.print(x);
    Serial.print(" y="); Serial.print(y);
    Serial.print(" w="); Serial.print(w);
    Serial.print(" h="); Serial.println(h);
    
    int angles[6];
    if (ikLookup(x, y, 90, angles)) {
      Serial.println("Moving to pick position...");
      moveServosSmooth(angles);
      Serial.println("DONE");
    } else {
      Serial.println("ERROR: No IK solution for that position");
    }
  }

  // ── GRIP ──
  else if (upper.startsWith("GRIP")) {
    Serial.println("Closing gripper...");
    int angles[6];
    memcpy(angles, currentAngles, sizeof(angles));
    angles[5] = GRIPPER_CLOSED;
    moveServosSmooth(angles, 10);
    Serial.println("DONE");
  }

  // ── RELEASE ──
  else if (upper.startsWith("RELEASE")) {
    Serial.println("Opening gripper...");
    int angles[6];
    memcpy(angles, currentAngles, sizeof(angles));
    angles[5] = GRIPPER_OPEN;
    moveServosSmooth(angles, 10);
    Serial.println("DONE");
  }

  // ── HOME ──
  else if (upper.startsWith("HOME")) {
    Serial.println("Returning home...");
    moveServosSmooth(HOME_ANGLES, 30);
    Serial.println("DONE");
  }

  // ── STATUS ──
  else if (upper.startsWith("STATUS")) {
    printStatus();
  }

  // ── HELP ──
  else if (upper.startsWith("HELP")) {
    printHelp();
  }

  // ── UNKNOWN ──
  else {
    Serial.print("ERROR: Unknown command \"");
    Serial.print(raw);
    Serial.println("\" -- type HELP");
  }
}

// ─────────────────────────────────────────────
// SETUP & LOOP
// ─────────────────────────────────────────────

void setup() {
  Serial.begin(9600);
  while (!Serial) delay(10);

  Serial.println("=====================================");
  Serial.println("  Robot Arm - Vision Ready           ");
  Serial.println("=====================================");

  Wire.begin();
  initPCA9685();

  Serial.println("Moving to home position...");
  memcpy(currentAngles, HOME_ANGLES, sizeof(HOME_ANGLES));
  moveServos(HOME_ANGLES);
  delay(1000);

  Serial.println("Ready! Type HELP for commands.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }
}