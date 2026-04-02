/**
 * ItsyBitsy M4 Firmware - Robot Arm Controller with PCA9685
 * Serial Monitor Control Version
 * v2.1 - Added coordinate translation:
 *   Incoming x=0 maps to IK table x=7 (arm base offset).
 *   All received x values are shifted by +X_OFFSET before IK lookup.
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

// Workspace limits in IK table space (after translation)
const float WS_X_MIN =  1.5f;
const float WS_X_MAX =  6.5f;
const float WS_Y_MIN =  0.6f;
const float WS_Y_MAX =  4.4f;

// ─────────────────────────────────────────────
// COORDINATE TRANSLATION
// ─────────────────────────────────────────────
// The vision system outputs coordinates where x=0 is directly
// in front of the arm base. The IK table was built with x=7
// at the arm base (L4 = 7 inch offset). This constant shifts
// all incoming x values into IK table space before lookup.
//
//   IK_x = received_x + X_OFFSET
//   e.g.  received 0.0 → IK 7.0
//         received 3.0 → IK 10.0
//         received 6.5 → IK 13.5

const float X_OFFSET = 7.0f;

// Translate received (x, y) into IK table coordinate space
void toIKSpace(float rx, float ry, float &ikx, float &iky) {
  ikx = rx + X_OFFSET;
  iky = ry;   // y needs no translation
}

// Compute base angle as the horizontal sweep angle to the target.
// Uses atan2(y, x) so the base rotates to point directly at the object.
// +90 offset maps 0 deg (straight ahead) to servo centre (90 deg).
int computeBaseAngle(float x, float y) {
  return (int)round(atan2(y, x) * 180.0f / PI) - 90;
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
    outputAngles[5] = currentAngles[5];  // preserve gripper state
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
  Serial.println("  MOVE x y           -- Move, x=0 is arm base (e.g. MOVE 3.5 2.0)");
  Serial.println("  PICK class x y w h -- From vision (e.g. PICK PLASTIC 3.5 2.0 100 80)");
  Serial.println("  GRIP               -- Close gripper");
  Serial.println("  RELEASE            -- Open gripper");
  Serial.println("  HOME               -- Return to home position");
  Serial.println("  STATUS             -- Show current servo angles");
  Serial.println("  HELP               -- Show this list");
  Serial.println("------------------------------------");
  Serial.println("  Note: x=0 maps to IK table x=7 (X_OFFSET=7)");
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

    Serial.print("Received ("); Serial.print(rx); Serial.print(", "); Serial.print(ry); Serial.println(")");
    Serial.print("IK space  ("); Serial.print(ikx); Serial.print(", "); Serial.print(iky); Serial.println(")");

    if (ikx < WS_X_MIN || ikx > WS_X_MAX || iky < WS_Y_MIN || iky > WS_Y_MAX) {
      Serial.print("WARNING: IK coords (");
      Serial.print(ikx); Serial.print(", "); Serial.print(iky);
      Serial.println(") outside reachable range");
    }

    int baseAngle = computeBaseAngle(ikx, iky);
    Serial.print("Base angle: "); Serial.print(baseAngle); Serial.println(" deg");

    int angles[6];
    if (!ikLookup(ikx, iky, baseAngle, angles)) {
      Serial.println("ERROR: No IK solution");
      return;
    }

    moveServosSmooth(angles);
    Serial.println("DONE");
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

    float ikx, iky;
    toIKSpace(rx, ry, ikx, iky);

    Serial.print("PICK: class="); Serial.print(classStr);
    Serial.print(" received=("); Serial.print(rx); Serial.print(", "); Serial.print(ry); Serial.print(")");
    Serial.print(" IK=("); Serial.print(ikx); Serial.print(", "); Serial.print(iky); Serial.println(")");

    if (ikx < WS_X_MIN || ikx > WS_X_MAX || iky < WS_Y_MIN || iky > WS_Y_MAX) {
      Serial.print("WARNING: IK coords (");
      Serial.print(ikx); Serial.print(", "); Serial.print(iky);
      Serial.println(") outside reachable range");
    }

    int baseAngle = computeBaseAngle(ikx, iky);
    Serial.print("Base angle: "); Serial.print(baseAngle); Serial.println(" deg");

    int angles[6];
    if (!ikLookup(ikx, iky, baseAngle, angles)) {
      Serial.println("ERROR: No IK solution for that position");
      return;
    }

    Serial.println("Moving to pick position...");
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
  Serial.println("  Robot Arm - Vision Ready  v2.1     ");
  Serial.println("  x=0 → IK x=7  (X_OFFSET=7)        ");
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