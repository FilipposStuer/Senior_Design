/**
 * ItsyBitsy M4 Firmware - Robot Arm Controller with PCA9685
 * Serial Monitor Control Version
 *
 * PURPOSE:
 *   Accepts X/Y coordinates typed into the Serial Monitor,
 *   looks up the corresponding servo angles from the IK table,
 *   and moves the arm to that position.
 *
 * COMMANDS (type into Serial Monitor, press Enter):
 *   MOVE x y        — Move arm to real-world coordinate (e.g. "MOVE 8.5 6.0")
 *   GRIP            — Close gripper
 *   RELEASE         — Open gripper
 *   HOME            — Return to home position
 *   STATUS          — Print current servo angles
 *   HELP            — Print command list
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "ik_table.h"   // Auto-generated from MATLAB

// PCA9685 CONFIGURATION
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

// PWM calibration — adjust these for your servos
uint16_t servoMin[6] = {220, 360, 60, 180, 60, 100};
uint16_t servoMax[6] = {600, 600, 600, 600, 600, 600};
#define SERVO_FREQ 50

// Gripper positions
const int GRIPPER_CLOSED = 20;
const int GRIPPER_OPEN   = 90;

// Home position
const int HOME_ANGLES[6] = {90, 90, 90, 90, 90, GRIPPER_OPEN};

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

/**
 * Find closest entry in IK table for given real-world X/Y (in inches/mm).
 * Stores resulting angles in outputAngles[6].
 * Returns true if a match was found.
 */
bool ikLookup(float x, float y, int outputAngles[6]) {
  int bestIdx  = -1;
  float minDist = 1e9;

  for (int i = 0; i < IK_TABLE_SIZE; i++) {
    IKEntry entry;
    memcpy_P(&entry, &IK_TABLE[i], sizeof(IKEntry));

    float dx   = entry.x - x;
    float dy   = entry.y - y;
    float dist = dx*dx + dy*dy;

    if (dist < minDist) {
      minDist  = dist;
      bestIdx  = i;
    }
  }

  if (bestIdx >= 0) {
    IKEntry best;
    memcpy_P(&best, &IK_TABLE[bestIdx], sizeof(IKEntry));
    memcpy(outputAngles, best.angles, 6 * sizeof(int));
    outputAngles[5] = GRIPPER_OPEN;  // Always approach with gripper open
    return true;
  }
  return false;
}

// ─────────────────────────────────────────────
// PRINT HELPERS
// ─────────────────────────────────────────────

void printStatus() {
  Serial.println("── Current Angles ──────────────────");
  const char* names[6] = {"Base      ", "Shoulder  ", "Elbow     ",
                           "Wrist     ", "Grip Rot  ", "Gripper   "};
  for (int i = 0; i < 6; i++) {
    Serial.print("  "); Serial.print(names[i]);
    Serial.print(": "); Serial.print(currentAngles[i]); Serial.println("°");
  }
  Serial.println("────────────────────────────────────");
}

void printHelp() {
  Serial.println("── Commands ────────────────────────");
  Serial.println("  MOVE x y   — Move to coordinate (e.g. MOVE 8.5 6.0)");
  Serial.println("  GRIP       — Close gripper");
  Serial.println("  RELEASE    — Open gripper");
  Serial.println("  HOME       — Return to home position");
  Serial.println("  STATUS     — Show current servo angles");
  Serial.println("  HELP       — Show this list");
  Serial.println("────────────────────────────────────");
}

// ─────────────────────────────────────────────
// COMMAND PROCESSING
// ─────────────────────────────────────────────

void processCommand(String raw) {
  raw.trim();
  if (raw.length() == 0) return;

  // Uppercase first word for command matching
  String upper = raw;
  upper.toUpperCase();

  // ── MOVE x y ──
  if (upper.startsWith("MOVE")) {
    float x, y;
    if (sscanf(raw.c_str(), "%*s %f %f", &x, &y) != 2) {
      Serial.println("ERROR: Usage is MOVE x y  (e.g. MOVE 8.5 6.0)");
      return;
    }

    Serial.print("Looking up IK for X="); Serial.print(x);
    Serial.print(" Y="); Serial.println(y);

    int angles[6];
    if (!ikLookup(x, y, angles)) {
      Serial.println("ERROR: No IK solution found for that position");
      return;
    }

    Serial.print("  Shoulder: "); Serial.print(angles[1]); Serial.println("°");
    Serial.print("  Elbow:    "); Serial.print(angles[2]); Serial.println("°");
    Serial.print("  Wrist:    "); Serial.print(angles[3]); Serial.println("°");
    Serial.println("Moving...");
    moveServosSmooth(angles);
    Serial.println("DONE");
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
    Serial.println("Returning to home position...");
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
    Serial.println("\" — type HELP for command list");
  }
}

// ─────────────────────────────────────────────
// SETUP & LOOP
// ─────────────────────────────────────────────

void setup() {
  Serial.begin(9600);
  while (!Serial) delay(10);

  Serial.println("╔══════════════════════════════════╗");
  Serial.println("║  Robot Arm - Serial Monitor Mode  ║");
  Serial.println("╚══════════════════════════════════╝");

  Wire.begin();
  initPCA9685();

  Serial.println("Moving to home position...");
  memcpy(currentAngles, HOME_ANGLES, sizeof(HOME_ANGLES));
  moveServos(HOME_ANGLES);
  delay(1000);

  Serial.println("Ready! Type HELP for commands.");
  Serial.println();
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }
}
