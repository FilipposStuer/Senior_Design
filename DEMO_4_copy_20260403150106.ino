/**
 * ItsyBitsy M4 Firmware - Robot Arm Controller with PCA9685
 * Serial Monitor Control Version
 * v2.2 - IK x-axis is now the Euclidean distance from the arm base
 *        to the target (sqrt(rx² + ry²)), shifted by X_OFFSET.
 *        The base servo still uses atan2(ry, rx) for rotation.
 *        Previously: IK_x = rx + X_OFFSET
 *        Now:        IK_x = sqrt(rx² + ry²) + X_OFFSET
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "ik_table4.h"
#include <math.h>

// ─────────────────────────────────────────────
// PCA9685 CONFIGURATION
// ─────────────────────────────────────────────

#define PCA9685_ADDR 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

const int CH_BASE        = 0;
const int CH_SHOULDER    = 1;
const int CH_ELBOW       = 2;
const int CH_WRIST       = 3;
const int CH_GRIPPER_ROT = 4;
const int CH_GRIPPER     = 5;
const int CHANNELS[6]    = {CH_BASE, CH_SHOULDER, CH_ELBOW, CH_WRIST, CH_GRIPPER_ROT, CH_GRIPPER};

uint16_t servoMin[6] = {130, 360, 60, 180, 60, 100};
uint16_t servoMax[6] = {600, 600, 600, 600, 600, 600};
#define SERVO_FREQ 50

const int GRIPPER_CLOSED = 20;
const int GRIPPER_OPEN   = 90;

const int HOME_ANGLES[6] = {90, 90, 90, 90, 90, GRIPPER_CLOSED};

// Workspace limits in IK table space (post-translation)
const float WS_X_MIN =  7.2f;
const float WS_X_MAX = 15.0f;
const float WS_Y_MIN =  0.6f;
const float WS_Y_MAX =  4.4f;

// ─────────────────────────────────────────────
// COORDINATE TRANSLATION
// ─────────────────────────────────────────────
// The vision system outputs (rx, ry) where rx is horizontal distance
// and ry is lateral offset from centre. The IK table's x-axis encodes
// radial reach from the arm base. We therefore feed it the Euclidean
// distance to the target, shifted by X_OFFSET so that a target directly
// in front at distance 0 maps to IK x=7 (the arm's physical base offset).
//
//   IK_x = sqrt(rx² + ry²) + X_OFFSET
//   IK_y = vertical height (unchanged)
//
//   e.g.  rx=0, ry=0   → IK_x = 0.0 + 7.0 = 7.0
//         rx=3, ry=4   → IK_x = 5.0 + 7.0 = 12.0
//         rx=6, ry=0   → IK_x = 6.0 + 7.0 = 13.0

const float X_OFFSET = 7.5f;

// Translate received (rx, ry) into IK table coordinate space.
// ikx = Euclidean reach distance + base offset.
// iky = vertical height, no translation needed.
void toIKSpace(float rx, float ry, float &ikx, float &iky) {
  ikx = sqrtf((rx + X_OFFSET) * (rx + X_OFFSET) + ry * ry);
  iky = ry;   // y (height) needs no translation
}

// Compute base angle as the horizontal sweep angle to the target.
// Uses atan2(ry, rx) so the base rotates to point directly at the object.
// +70 offset maps 0 deg (straight ahead) to servo centre.
int computeBaseAngle(float rx, float ry) {
  return (int)round(atan2f(ry, rx) * 180.0f / PI) + 70;
}

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
    float dy   = IK_TABLE[i].y - 1;
    float dist = dx*dx + dy*dy;
    if (dist < minDist) {
      minDist = dist;
      bestIdx = i;
    }
  }

  if (bestIdx >= 0) {
    memcpy(outputAngles, IK_TABLE[bestIdx].angles, 6 * sizeof(int));
    outputAngles[0] = constrain(baseAngle, 0, 180);
    outputAngles[5] = currentAngles[5];
    Serial.print("  Matched table entry: x="); Serial.print(IK_TABLE[bestIdx].x);
    Serial.print(" y="); Serial.println(IK_TABLE[bestIdx].y);
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
  Serial.println("  MOVE x y           -- Move to position (e.g. MOVE 3.5 2.0)");
  Serial.println("  PICK class x y w h -- From vision (e.g. PICK PLASTIC 3.5 2.0 100 80)");
  Serial.println("  GRIP               -- Close gripper");
  Serial.println("  RELEASE            -- Open gripper");
  Serial.println("  HOME               -- Return to home position");
  Serial.println("  STATUS             -- Show current servo angles");
  Serial.println("  HELP               -- Show this list");
  Serial.println("------------------------------------");
  Serial.println("  Note: IK_x = sqrt(rx^2 + ry^2) + X_OFFSET(7)");
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

    float rx = coords.substring(0, sp1).toFloat();
    float ry = coords.substring(sp1 + 1).toFloat();

    float ikx, iky;
    toIKSpace(rx, ry, ikx, iky);

    Serial.print("Received  ("); Serial.print(rx); Serial.print(", "); Serial.print(ry); Serial.println(")");
    Serial.print("Distance  "); Serial.print(sqrtf(rx*rx + ry*ry)); Serial.println(" in");
    Serial.print("IK space  ("); Serial.print(ikx); Serial.print(", "); Serial.print(iky); Serial.println(")");

    if (ikx < WS_X_MIN || ikx > WS_X_MAX || iky < WS_Y_MIN || iky > WS_Y_MAX) {
      Serial.print("WARNING: IK coords (");
      Serial.print(ikx); Serial.print(", "); Serial.print(iky);
      Serial.println(") outside reachable range");
    }

    int baseAngle = computeBaseAngle(rx, ry);
    Serial.print("Base angle: "); Serial.print(baseAngle); Serial.println(" deg");

    int angles[6];
    if (!ikLookup(ikx, iky, baseAngle, angles)) {
      Serial.println("ERROR: No IK solution");
      return;
    }

    moveServosSmooth(angles);
    Serial.println("DONE");
    Serial1.println("DONE");
  }

  // ── PICK (from vision) ──
  else if (upper.startsWith("PICK")) {
    String rest = raw.substring(4);
    rest.trim();

    int firstSpace = rest.indexOf(' ');
    if (firstSpace == -1) {
      Serial.println("ERROR: Invalid PICK format");
      return;
    }

    String classStr = rest.substring(0, firstSpace);
    String coords   = rest.substring(firstSpace + 1);
    coords.trim();

    int sp1 = coords.indexOf(' ');
    int sp2 = coords.indexOf(' ', sp1 + 1);
    int sp3 = coords.indexOf(' ', sp2 + 1);

    if (sp1 == -1 || sp2 == -1 || sp3 == -1) {
      Serial.println("ERROR: Need 4 numbers after class (x y w h)");
      return;
    }

    float rx = coords.substring(0, sp1).toFloat();
    float ry = coords.substring(sp1 + 1, sp2).toFloat();
    int   w  = (int)coords.substring(sp2 + 1, sp3).toFloat();
    int   h  = (int)coords.substring(sp3 + 1).toFloat();

    // Use real rx, ry for Euclidean distance; iky fixed at 1.0 (floor height)
    float ikx, iky;
    toIKSpace(rx, ry, ikx, iky);
    iky = 1.0f;  // override height to floor pickup level

    int baseAngle = computeBaseAngle(rx, ry);

    Serial.print("PICK: class="); Serial.print(classStr);
    Serial.print(" received=("); Serial.print(rx); Serial.print(", "); Serial.print(ry); Serial.print(")");
    Serial.print(" dist="); Serial.print(ikx - X_OFFSET);
    Serial.print(" IK=("); Serial.print(ikx); Serial.print(", "); Serial.print(iky); Serial.println(")");

    if (ikx < WS_X_MIN || ikx > WS_X_MAX || iky < WS_Y_MIN || iky > WS_Y_MAX) {
      Serial.print("WARNING: IK coords (");
      Serial.print(ikx); Serial.print(", "); Serial.print(iky);
      Serial.println(") outside reachable range");
    }

    Serial.print("Base angle: "); Serial.print(baseAngle); Serial.println(" deg");

    int angles[6];
    if (!ikLookup(ikx, iky, baseAngle, angles)) {
      Serial.println("ERROR: No IK solution for that position");
      return;
    }

    Serial.println("Moving to pick position...");
    moveServosSmooth(angles);

    Serial.println("Closing gripper...");
    int gripAngles[6];
    memcpy(gripAngles, currentAngles, sizeof(gripAngles));
    gripAngles[5] = GRIPPER_CLOSED;
    moveServosSmooth(gripAngles, 10);

    Serial.println("DONE");
    Serial1.println("DONE");
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
  Serial.println("  Robot Arm - Vision Ready  v2.2     ");
  Serial.println("  IK_x = sqrt(rx^2+ry^2) + X_OFFSET  ");
  Serial.println("=====================================");

  Wire.begin();
  Serial1.begin(9600);
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