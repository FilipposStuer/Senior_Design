/**
 * ItsyBitsy M4 Firmware - Robot Arm Controller with PCA9685
 * Serial Monitor Control Version
 * v2.0 - Fixes applied to original Demo_march_24.ino:
 *   1. Removed PROGMEM / memcpy_P  (ItsyBitsy M4 is ARM, not AVR)
 *   2. Replaced sscanf %f (unreliable on Arduino ARM) with toFloat()
 *   3. Added optional base angle parameter to MOVE command
 *   4. Added workspace boundary warning
 *
 * COMMANDS (type into Serial Monitor, press Enter):
 *   MOVE x y        -- Move arm (e.g. "MOVE 8.5 6.0")
 *   MOVE x y base   -- Move arm and set base angle (e.g. "MOVE 8.5 6.0 45")
 *                      x    = radial distance from base (cm)
 *                      y    = height (cm)
 *                      base = base servo angle in degrees (0-180, default 90)
 *   GRIP            -- Close gripper
 *   RELEASE         -- Open gripper
 *   HOME            -- Return to home position
 *   STATUS          -- Print current servo angles
 *   HELP            -- Print command list
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
uint16_t servoMin[6] = {220, 360, 60, 180, 60, 100};
uint16_t servoMax[6] = {600, 600, 600, 600, 600, 600};
#define SERVO_FREQ 50

// Gripper positions
const int GRIPPER_CLOSED = 20;
const int GRIPPER_OPEN   = 90;

// Home position
const int HOME_ANGLES[6] = {90, 90, 90, 90, 90, GRIPPER_OPEN};

// Workspace limits from ik_table.h
const float WS_X_MIN =  4.0f;
const float WS_X_MAX = 12.5f;
const float WS_Y_MIN =  1.0f;
const float WS_Y_MAX = 12.0f;

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

// All joints arrive at the same time by stepping at the rate of
// the joint with the largest angular change.
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
// Fix 1: Removed PROGMEM / memcpy_P.
//   The ItsyBitsy M4 is ARM Cortex-M4. On ARM the compiler places
//   const arrays in Flash (.rodata) automatically -- no PROGMEM needed.
//   Direct array access replaces memcpy_P throughout.
//
// x = radial distance from base (cm)  -- matches table entry.x
// y = height (cm)                     -- matches table entry.y
// baseAngle = base servo angle (0-180 deg), passed in from the command

bool ikLookup(float x, float y, int baseAngle, int outputAngles[6]) {
  int   bestIdx = -1;
  float minDist = 1e9f;

  for (int i = 0; i < IK_TABLE_SIZE; i++) {
    // Fix 1: direct array access instead of memcpy_P
    float dx   = IK_TABLE[i].x - x;
    float dy   = IK_TABLE[i].y - y;
    float dist = dx*dx + dy*dy;

    if (dist < minDist) {
      minDist = dist;
      bestIdx = i;
    }
  }

  if (bestIdx >= 0) {
    // Fix 1: direct array access instead of memcpy_P
    memcpy(outputAngles, IK_TABLE[bestIdx].angles, 6 * sizeof(int));
    outputAngles[0] = constrain(baseAngle, 0, 180); // override base with user value
    outputAngles[5] = GRIPPER_OPEN;                 // always approach with gripper open
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
  Serial.println("  MOVE x y        -- Move (e.g. MOVE 8.5 6.0)");
  Serial.println("  MOVE x y base   -- Move + set base angle (e.g. MOVE 8.5 6.0 45)");
  Serial.println("                     x=radial dist (cm), y=height (cm), base=0-180 deg");
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

// Fix 2: Replaced sscanf with toFloat().
//   sscanf with %f is unreliable on many Arduino/ARM boards -- it silently
//   returns 0 parsed values even for valid float strings. Arduino's
//   String::toFloat() is reliable and used here instead.

void processCommand(String raw) {
  raw.trim();
  if (raw.length() == 0) return;

  String upper = raw;
  upper.toUpperCase();

  // ── MOVE x y [base] ──
  if (upper.startsWith("MOVE")) {
    String coords = raw.substring(4);
    coords.trim();

    // Parse first token: x
    int sp1 = coords.indexOf(' ');
    if (sp1 == -1) {
      Serial.println("ERROR: Usage is MOVE x y  (e.g. MOVE 8.5 6.0)");
      return;
    }
    String xStr = coords.substring(0, sp1);
    coords = coords.substring(sp1 + 1);
    coords.trim();

    // Parse second token: y
    int sp2 = coords.indexOf(' ');
    String yStr, baseStr;
    int baseAngle = 90; // default base angle

    if (sp2 == -1) {
      // Only two values provided -- use default base angle
      yStr = coords;
    } else {
      // Three values provided -- third is base angle
      yStr    = coords.substring(0, sp2);
      baseStr = coords.substring(sp2 + 1);
      baseStr.trim();
      baseAngle = constrain((int)baseStr.toFloat(), 0, 180);
    }

    float x = xStr.toFloat();
    float y = yStr.toFloat();

    // toFloat() returns 0.0 on failure; check that the string was actually '0'
    if (x == 0.0f && xStr.charAt(0) != '0') { Serial.println("ERROR: 'x' is not a valid number"); return; }
    if (y == 0.0f && yStr.charAt(0) != '0') { Serial.println("ERROR: 'y' is not a valid number"); return; }

    // Fix 4: Workspace boundary warning
    bool outOfRange = false;
    if (x < WS_X_MIN || x > WS_X_MAX) {
      Serial.print("WARNING: x="); Serial.print(x, 2);
      Serial.print(" is outside table range ["); Serial.print(WS_X_MIN);
      Serial.print(", "); Serial.print(WS_X_MAX); Serial.println("] -- using nearest point");
      outOfRange = true;
    }
    if (y < WS_Y_MIN || y > WS_Y_MAX) {
      Serial.print("WARNING: y="); Serial.print(y, 2);
      Serial.print(" is outside table range ["); Serial.print(WS_Y_MIN);
      Serial.print(", "); Serial.print(WS_Y_MAX); Serial.println("] -- using nearest point");
      outOfRange = true;
    }
    if (outOfRange) Serial.println("  Result may be imprecise.");

    Serial.print("Looking up IK for x="); Serial.print(x);
    Serial.print(" y="); Serial.print(y);
    Serial.print(" base="); Serial.println(baseAngle);

    int angles[6];
    if (!ikLookup(x, y, baseAngle, angles)) {
      Serial.println("ERROR: No IK solution found for that position");
      return;
    }

    Serial.print("  Base:     "); Serial.print(angles[0]); Serial.println(" deg");
    Serial.print("  Shoulder: "); Serial.print(angles[1]); Serial.println(" deg");
    Serial.print("  Elbow:    "); Serial.print(angles[2]); Serial.println(" deg");
    Serial.print("  Wrist:    "); Serial.print(angles[3]); Serial.println(" deg");
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
    Serial.println("\" -- type HELP for command list");
  }
}

// ─────────────────────────────────────────────
// SETUP & LOOP
// ─────────────────────────────────────────────

void setup() {
  Serial.begin(9600);
  while (!Serial) delay(10);

  Serial.println("=====================================");
  Serial.println("  Robot Arm -- Serial Monitor Mode   ");
  Serial.println("             v2.0                    ");
  Serial.println("=====================================");

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
