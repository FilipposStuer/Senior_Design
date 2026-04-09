/**
 * ItsyBitsy M4 Firmware - Robot Arm Controller with PCA9685
 * Serial Monitor Control Version
 * v2.3 - PLASTIC/PAPER/CARDBOARD trigger pick-then-drop-to-bin sequence.
 *        All other classes (biodegradable, glass, metal) do a normal pick and stop.
 *        Bin position: rx=0, ry=14.375 (14+3/8), iky=7.0 (elevated drop)
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

const int GRIPPER_CLOSED = 90;   // physically closes the gripper
const int GRIPPER_OPEN   = 20;   // physically opens the gripper

const int HOME_ANGLES[6] = {90, 90, 90, 90, 90, GRIPPER_OPEN};

// Workspace limits in IK table space (post-translation)
const float WS_X_MIN =  7.2f;
const float WS_X_MAX = 15.0f;
const float WS_Y_MIN =  0.6f;
const float WS_Y_MAX =  8.4f;

// ─────────────────────────────────────────────
// BIN ANGLES (direct servo angles per class)
// ─────────────────────────────────────────────
const int BIN_PAPER[6]     = { 50,  10,  90,  90, 90, 20};
const int BIN_PLASTIC[6]   = { 25,  65, 110, 130, 90, 20};
const int BIN_CARDBOARD[6] = {118,   0,  90,  90, 90, 20};

// ─────────────────────────────────────────────
// COORDINATE TRANSLATION
// ─────────────────────────────────────────────

const float X_OFFSET = 7.5f;

void toIKSpace(float rx, float ry, float &ikx, float &iky) {
  ikx = sqrtf((rx + 3) * (rx + 3) + ry * ry) + 2.0f;
  iky = 3.0f;   // default floor height; callers may override
}

int computeBaseAngle(float rx, float ry) {
  return (int)round(atan2f(ry, X_OFFSET - rx) * 180.0f / PI);
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
    delay(15);
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
    outputAngles[5] = currentAngles[5];
    Serial.print("  Matched table entry: x="); Serial.print(IK_TABLE[bestIdx].x);
    Serial.print(" y="); Serial.println(IK_TABLE[bestIdx].y);
    return true;
  }
  return false;
}

// ─────────────────────────────────────────────
// DROP TO BIN
// ─────────────────────────────────────────────
// Moves arm directly to the hardcoded bin angles for the given class,
// opens the gripper to release the object, then returns HOME.

void dropToBin(String classStr) {
  const int* binAngles = nullptr;

  String upper = classStr;
  upper.toUpperCase();

  if      (upper == "PLASTIC")   binAngles = BIN_PLASTIC;
  else if (upper == "PAPER")     binAngles = BIN_PAPER;
  else if (upper == "CARDBOARD") binAngles = BIN_CARDBOARD;
  else {
    Serial.print("ERROR: Unknown bin class '"); Serial.print(classStr); Serial.println("'");
    return;
  }

  // Copy bin angles, keep gripper closed while travelling
  int angles[6];
  memcpy(angles, binAngles, sizeof(angles));
  angles[5] = GRIPPER_CLOSED;

  Serial.print("Moving to "); Serial.print(classStr); Serial.println(" bin...");
  moveServosSmooth(angles);

  // Release object
  Serial.println("Releasing into bin...");
  angles[5] = GRIPPER_OPEN;
  moveServosSmooth(angles, 10);

  // Jiggle gripper to ensure object falls into bin
  Serial.println("Jiggling gripper...");
  for (int i = 0; i < 4; i++) {
    angles[5] = GRIPPER_CLOSED;
    moveServosSmooth(angles, 10);
    angles[5] = GRIPPER_OPEN;
    moveServosSmooth(angles, 10);
  }

  // Return home
  Serial.println("Returning home...");
  int home[6] = {90, 90, 90, 90, 90, GRIPPER_OPEN};
  moveServosSmooth(home, 30);

  Serial.println("DONE");
  Serial1.println("DONE");
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
  Serial.println("  PICK class x y w h -- From vision");
  Serial.println("    PLASTIC/PAPER/CARDBOARD -> pick at coords, drop to class bin, HOME");
  Serial.println("    BIODEGRADABLE/GLASS/METAL -> pick at coords and stop");
  Serial.println("  GRIP               -- Close gripper");
  Serial.println("  RELEASE            -- Open gripper");
  Serial.println("  HOME               -- Return to home position");
  Serial.println("  STATUS             -- Show current servo angles");
  Serial.println("  HELP               -- Show this list");
  Serial.println("------------------------------------");
  Serial.println("  Note: IK_x = sqrt((rx+3)^2 + ry^2) + 2.0");
  Serial.println("  Bin:  rx=0  ry=14.375  iky=7.0 (elevated)");
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

    String classStr  = rest.substring(0, firstSpace);
    String upperClass = classStr;
    upperClass.toUpperCase();

    // Parse coords (shared by both branches)
    String coords = rest.substring(firstSpace + 1);
    coords.trim();

    int sp1 = coords.indexOf(' ');
    int sp2 = coords.indexOf(' ', sp1 + 1);
    int sp3 = coords.indexOf(' ', sp2 + 1);
    int sp4 = coords.indexOf(' ', sp3 + 1);

    if (sp1 == -1 || sp2 == -1 || sp3 == -1) {
      Serial.println("ERROR: Need 4 numbers after class (x y w h)");
      return;
    }

    float rx = coords.substring(0, sp1).toFloat();
    float ry = coords.substring(sp1 + 1, sp2).toFloat();
    int   w  = (int)coords.substring(sp2 + 1, sp3).toFloat();
    int   h  = (int)coords.substring(sp3 + 1, sp4 == -1 ? coords.length() : sp4).toFloat();
    int   gripRot = (sp4 == -1) ? 90 : coords.substring(sp4 + 1).toInt();

    float ikx, iky;
    toIKSpace(rx, ry, ikx, iky);
    iky = 3.0f;   // floor pickup height

    int baseAngle = computeBaseAngle(rx, ry);

    Serial.print("PICK: class="); Serial.print(classStr);
    Serial.print(" received=("); Serial.print(rx); Serial.print(", "); Serial.print(ry); Serial.print(")");
    Serial.print(" IK=("); Serial.print(ikx); Serial.print(", "); Serial.print(iky); Serial.println(")");
    Serial.print("Base angle: "); Serial.print(baseAngle); Serial.println(" deg");

    if (ikx < WS_X_MIN || ikx > WS_X_MAX || iky < WS_Y_MIN || iky > WS_Y_MAX) {
      Serial.print("WARNING: IK coords (");
      Serial.print(ikx); Serial.print(", "); Serial.print(iky);
      Serial.println(") outside reachable range");
    }

    int angles[6];
    if (!ikLookup(ikx, iky, baseAngle, angles)) {
      Serial.println("ERROR: No IK solution for that position");
      return;
    }

    // ── PLASTIC / PAPER / CARDBOARD: pick then drop to bin ──
    if (upperClass == "PLASTIC" || upperClass == "PAPER" || upperClass == "CARDBOARD") {
      Serial.println("Moving to pick position...");
      angles[4] = constrain(gripRot, 0, 180);   // apply gripper rotation
      moveServosSmooth(angles);

      // Close gripper to grab object
      angles[5] = GRIPPER_CLOSED;
      moveServosSmooth(angles, 10);

      // Return home first (carrying the object)
      Serial.println("Returning home with object...");
      int home[6] = {90, 90, 90, 90, 90, GRIPPER_CLOSED};
      moveServosSmooth(home, 30);
      delay(500);   // wait for servos to physically reach home before continuing

      // Now carry it to the bin
      dropToBin(classStr);   // handles travel, release, and HOME internally
      return;
    }

    // ── All other classes (biodegradable, glass, metal): ignore ──
    Serial.print("Ignoring class: "); Serial.println(classStr);
    Serial1.println("DONE");
    return;
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
    int angles[6] = {90, 90, 90, 90, 90, 90};
    moveServosSmooth(angles, 30);
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
  Serial.println("  Robot Arm - Vision Ready  v2.3     ");
  Serial.println("  PLASTIC/PAPER/CARDBOARD -> bin      ");
  Serial.println("  BIO/GLASS/METAL -> pick & stop      ");
  Serial.println("=====================================");

  Wire.begin();
  Serial1.begin(9600);
  initPCA9685();

  Serial.println("Moving to home position...");
  memcpy(currentAngles, HOME_ANGLES, sizeof(HOME_ANGLES));
  moveServos(HOME_ANGLES);
  delay(100);

  Serial.println("Ready! Type HELP for commands.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }
}