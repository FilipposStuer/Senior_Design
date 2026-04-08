/**
 * Robot Arm - Position Calibration Tool
 * 
 * Manually jog each servo to position the arm anywhere in space,
 * then save the angles. Saved positions are printed as IK table
 * entries ready to copy into ik_table.h.
 *
 * Commands:
 *   B+  / B-   -- Jog base        (+/- step)
 *   S+  / S-   -- Jog shoulder    (+/- step)
 *   E+  / E-   -- Jog elbow       (+/- step)
 *   W+  / W-   -- Jog wrist       (+/- step)
 *   R+  / R-   -- Jog gripper rot (+/- step)
 *   G+  / G-   -- Jog gripper     (+/- step)
 *   STEP n     -- Set jog step size (default 1 deg)
 *   SET a b c d e f  -- Set all 6 angles directly
 *   HOME       -- Go to home position
 *   STATUS     -- Print current angles
 *   SAVE x y   -- Save current angles as IK entry at (x, y)
 *   LIST       -- Print all saved entries
 *   CLEAR      -- Clear all saved entries
 *   HELP       -- Show this list
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

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

const int HOME_ANGLES[6] = {90, 90, 90, 90, 90, 20};

// ─────────────────────────────────────────────
// SAVED POSITIONS
// ─────────────────────────────────────────────

#define MAX_SAVED 64

struct SavedPos {
  float x;
  float y;
  int   angles[6];
};

SavedPos saved[MAX_SAVED];
int      savedCount = 0;

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────

int currentAngles[6];
int jogStep = 1;   // degrees per jog command

// ─────────────────────────────────────────────
// PCA9685 FUNCTIONS
// ─────────────────────────────────────────────

void initPCA9685() {
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  delay(10);
}

void setServoAngle(int ch, int angle) {
  angle = constrain(angle, 0, 180);
  uint16_t pulse = map(angle, 0, 180, servoMin[ch], servoMax[ch]);
  pwm.setPWM(CHANNELS[ch], 0, pulse);
}

void applyAngles() {
  for (int i = 0; i < 6; i++)
    setServoAngle(i, currentAngles[i]);
}

// ─────────────────────────────────────────────
// PRINT HELPERS
// ─────────────────────────────────────────────

void printStatus() {
  const char* names[6] = {"Base     ", "Shoulder ", "Elbow    ",
                           "Wrist    ", "Grip Rot ", "Gripper  "};
  Serial.println("-- Current Angles ------------------");
  for (int i = 0; i < 6; i++) {
    Serial.print("  "); Serial.print(names[i]);
    Serial.print(": "); Serial.print(currentAngles[i]);
    Serial.println(" deg");
  }
  Serial.print("  Jog step: "); Serial.print(jogStep); Serial.println(" deg");
  Serial.println("------------------------------------");
}

void printEntry(int idx) {
  Serial.print("  {");
  Serial.print(saved[idx].x); Serial.print(", ");
  Serial.print(saved[idx].y); Serial.print(", {");
  for (int j = 0; j < 6; j++) {
    Serial.print(saved[idx].angles[j]);
    if (j < 5) Serial.print(", ");
  }
  Serial.println("}}");
}

void printList() {
  if (savedCount == 0) {
    Serial.println("No saved positions.");
    return;
  }
  Serial.println("-- Saved Positions (IK table format) --");
  for (int i = 0; i < savedCount; i++) {
    Serial.print("["); Serial.print(i); Serial.print("] ");
    printEntry(i);
  }
  Serial.println("---------------------------------------");
  Serial.println("-- Copy-paste block for ik_table.h: --");
  for (int i = 0; i < savedCount; i++) {
    Serial.print("  {");
    Serial.print(saved[i].x); Serial.print("f, ");
    Serial.print(saved[i].y); Serial.print("f, {");
    Serial.print(saved[i].angles[0]); Serial.print(", ");
    Serial.print(saved[i].angles[1]); Serial.print(", ");
    Serial.print(saved[i].angles[2]); Serial.print(", ");
    Serial.print(saved[i].angles[3]); Serial.print(", ");
    Serial.print(saved[i].angles[4]); Serial.print(", ");
    Serial.print(saved[i].angles[5]);
    Serial.println("}},");
  }
  Serial.println("---------------------------------------");
}

void printHelp() {
  Serial.println("-- Commands ------------------------");
  Serial.println("  B+/B-        Jog base");
  Serial.println("  S+/S-        Jog shoulder");
  Serial.println("  E+/E-        Jog elbow");
  Serial.println("  W+/W-        Jog wrist");
  Serial.println("  R+/R-        Jog gripper rotation");
  Serial.println("  G+/G-        Jog gripper");
  Serial.println("  STEP n       Set jog step size in degrees");
  Serial.println("  SET a b c d e f   Set all 6 angles");
  Serial.println("  HOME         Go to home position");
  Serial.println("  STATUS       Show current angles");
  Serial.println("  SAVE x y     Save current angles at IK coords (x, y)");
  Serial.println("  LIST         Print all saved entries");
  Serial.println("  CLEAR        Clear all saved entries");
  Serial.println("  HELP         Show this list");
  Serial.println("------------------------------------");
}

// ─────────────────────────────────────────────
// JOG HELPER
// ─────────────────────────────────────────────

void jog(int ch, int dir) {
  currentAngles[ch] = constrain(currentAngles[ch] + dir * jogStep, 0, 180);
  setServoAngle(ch, currentAngles[ch]);
  const char* names[6] = {"Base", "Shoulder", "Elbow", "Wrist", "GripRot", "Gripper"};
  Serial.print(names[ch]); Serial.print(" -> ");
  Serial.print(currentAngles[ch]); Serial.println(" deg");
}

// ─────────────────────────────────────────────
// COMMAND PROCESSING
// ─────────────────────────────────────────────

void processCommand(String raw) {
  raw.trim();
  if (raw.length() == 0) return;

  String upper = raw;
  upper.toUpperCase();

  // ── Jog commands ──
  if (upper == "B+")  { jog(0,  1); return; }
  if (upper == "B-")  { jog(0, -1); return; }
  if (upper == "S+")  { jog(1,  1); return; }
  if (upper == "S-")  { jog(1, -1); return; }
  if (upper == "E+")  { jog(2,  1); return; }
  if (upper == "E-")  { jog(2, -1); return; }
  if (upper == "W+")  { jog(3,  1); return; }
  if (upper == "W-")  { jog(3, -1); return; }
  if (upper == "R+")  { jog(4,  1); return; }
  if (upper == "R-")  { jog(4, -1); return; }
  if (upper == "G+")  { jog(5,  1); return; }
  if (upper == "G-")  { jog(5, -1); return; }

  // ── STEP n ──
  if (upper.startsWith("STEP")) {
    String val = raw.substring(4);
    val.trim();
    int s = val.toInt();
    if (s < 1 || s > 90) {
      Serial.println("ERROR: Step must be 1-90");
      return;
    }
    jogStep = s;
    Serial.print("Jog step set to "); Serial.print(jogStep); Serial.println(" deg");
    return;
  }

  // ── SET a b c d e f ──
  if (upper.startsWith("SET")) {
    String rest = raw.substring(3);
    rest.trim();
    int vals[6];
    int idx = 0;
    while (idx < 6 && rest.length() > 0) {
      int sp = rest.indexOf(' ');
      if (sp == -1) {
        vals[idx++] = rest.toInt();
        rest = "";
      } else {
        vals[idx++] = rest.substring(0, sp).toInt();
        rest = rest.substring(sp + 1);
        rest.trim();
      }
    }
    if (idx < 6) {
      Serial.println("ERROR: Need 6 angles (base shoulder elbow wrist griprot gripper)");
      return;
    }
    for (int i = 0; i < 6; i++) currentAngles[i] = constrain(vals[i], 0, 180);
    applyAngles();
    printStatus();
    return;
  }

  // ── SAVE x y ──
  if (upper.startsWith("SAVE")) {
    String rest = raw.substring(4);
    rest.trim();
    int sp = rest.indexOf(' ');
    if (sp == -1) {
      Serial.println("ERROR: Usage is SAVE x y");
      return;
    }
    float x = rest.substring(0, sp).toFloat();
    float y = rest.substring(sp + 1).toFloat();

    if (savedCount >= MAX_SAVED) {
      Serial.println("ERROR: Max saved positions reached (64)");
      return;
    }

    saved[savedCount].x = x;
    saved[savedCount].y = y;
    memcpy(saved[savedCount].angles, currentAngles, sizeof(currentAngles));
    Serial.print("Saved ["); Serial.print(savedCount); Serial.print("] at (");
    Serial.print(x); Serial.print(", "); Serial.print(y); Serial.println("):");
    printEntry(savedCount);
    savedCount++;
    return;
  }

  // ── LIST ──
  if (upper.startsWith("LIST")) {
    printList();
    return;
  }

  // ── CLEAR ──
  if (upper.startsWith("CLEAR")) {
    savedCount = 0;
    Serial.println("All saved positions cleared.");
    return;
  }

  // ── HOME ──
  if (upper.startsWith("HOME")) {
    memcpy(currentAngles, HOME_ANGLES, sizeof(HOME_ANGLES));
    applyAngles();
    Serial.println("Home.");
    printStatus();
    return;
  }

  // ── STATUS ──
  if (upper.startsWith("STATUS")) {
    printStatus();
    return;
  }

  // ── HELP ──
  if (upper.startsWith("HELP")) {
    printHelp();
    return;
  }

  Serial.print("ERROR: Unknown command \"");
  Serial.print(raw);
  Serial.println("\" -- type HELP");
}

// ─────────────────────────────────────────────
// SETUP & LOOP
// ─────────────────────────────────────────────

void setup() {
  Serial.begin(9600);
  while (!Serial) delay(10);

  Serial.println("=====================================");
  Serial.println("  Robot Arm - Calibration Tool       ");
  Serial.println("=====================================");

  Wire.begin();
  initPCA9685();

  memcpy(currentAngles, HOME_ANGLES, sizeof(HOME_ANGLES));
  applyAngles();
  delay(500);

  Serial.println("Ready. Type HELP for commands.");
  printStatus();
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }
}
