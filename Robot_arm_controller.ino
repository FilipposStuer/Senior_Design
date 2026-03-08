/**
 * ItsyBitsy M4 Firmware - Robot Arm Controller with PCA9685
 * 
 * PURPOSE:
 *   This code runs on the ItsyBitsy M4 microcontroller and controls a 6-DOF robot arm
 *   using a PCA9685 PWM driver. It receives commands from a Raspberry Pi over USB serial
 *   and moves the arm to pick waste items and place them in appropriate bins.
 * 
 * COMMUNICATION WITH RASPBERRY PI:
 *   - Receives: PICK class cx cy bw bh  (initial command to start picking)
 *   - Receives: POS cx cy              (continuous position updates for PID)
 *   - Receives: TARGET_REACHED          (object is now covered by the arm)
 *   - Sends: DONE or ERROR              (confirmation after full sequence)
 * 
 * HARDWARE:
 *   - ItsyBitsy M4 (microcontroller)
 *   - PCA9685 16-channel PWM driver (I2C address 0x40 default)
 *   - 6 servo motors (connected to PCA9685 channels 0-5)
 * 
 * CONTROL STRATEGY:
 *   - Inverse kinematics using pre-calculated lookup tables (from simulation)
 *   - Closed-loop PID control using visual feedback from Pi
 *   - Finite state machine for reliable sequence execution
 */

#include <Wire.h>                    // I2C communication library
#include <Adafruit_PWMServoDriver.h> // Library for PCA9685 PWM driver

//  PCA9685 CONFIGURATION 
// I2C address of PCA9685 (default 0x40, can be changed with jumpers)
#define PCA9685_ADDR 0x40
// Create PWM driver object
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

// Servo channels on PCA9685 (0-15)
const int CH_BASE = 0;           // Base rotation (left/right)
const int CH_SHOULDER = 1;       // Shoulder (up/down)
const int CH_ELBOW = 2;          // Elbow (forward/back)
const int CH_WRIST = 3;          // Wrist (angle adjustment)
const int CH_GRIPPER_ROT = 4;    // Gripper rotation (twist)
const int CH_GRIPPER = 5;        // Gripper open/close

// Array mapping logical servo indices to physical channels
const int CHANNELS[6] = {CH_BASE, CH_SHOULDER, CH_ELBOW, CH_WRIST, CH_GRIPPER_ROT, CH_GRIPPER};

// PWM values for servos (0-4096 range on PCA9685)
// These need calibration for your specific servos!
#define SERVO_MIN 150   // Minimum pulse width (0 degrees) - ADJUST THIS
#define SERVO_MAX 600   // Maximum pulse width (180 degrees) - ADJUST THIS
#define SERVO_FREQ 50   // Frequency for analog servos (50Hz standard)

// Individual calibration for each servo (if they have different ranges)
uint16_t servoMin[6] = {150, 150, 150, 150, 150, 150};  // Min values per channel
uint16_t servoMax[6] = {600, 600, 600, 600, 600, 600};  // Max values per channel

// ROBOT ARM CONFIGURATION 
// Home position (rest position) - all servos at 90 degrees
const int HOME_ANGLES[6] = {90, 90, 90, 90, 90, 90};

// Gripper closed position (adjust based on your gripper mechanism)
const int GRIPPER_CLOSED = 20;   // Angle when gripper is fully closed
const int GRIPPER_OPEN = 90;     // Angle when gripper is fully open

// BIN POSITIONS 
// Structure to hold bin position data
struct BinPosition {
  const char* name;          // Bin name (PAPER, PLASTIC, GENERAL)
  int angles[6];             // Servo angles to reach this bin [base, shoulder, elbow, wrist, grip_rot, gripper]
};

// Bin definitions - THESE ARE EXAMPLE VALUES! Replace with your actual calculated angles
BinPosition BIN_PAPER = {"PAPER", {30, 45, 60, 75, 90, GRIPPER_OPEN}};
BinPosition BIN_PLASTIC = {"PLASTIC", {60, 45, 60, 75, 90, GRIPPER_OPEN}};
BinPosition BIN_GENERAL = {"GENERAL", {90, 45, 60, 75, 90, GRIPPER_OPEN}};

// PID CONTROL PARAMETERS 
float Kp = 1.0;    // Proportional gain - reacts to current error
float Ki = 0.1;    // Integral gain - reacts to accumulated error (eliminates steady-state error)
float Kd = 0.05;   // Derivative gain - reacts to rate of change (dampens oscillations)
const int TARGET_TOLERANCE = 10;  // Pixels - how close is "close enough"

// INVERSE KINEMATICS LOOKUP TABLE
// Structure for lookup table entries
struct IKEntry {
  int x;              // X coordinate of object center (pixels from camera)
  int y;              // Y coordinate of object center (pixels from camera)
  int w;              // Width of bounding box (pixels - used to estimate depth/distance)
  int angles[6];      // Pre-calculated servo angles for this position
};

/**
 * IMPORTANT: This is a SAMPLE lookup table!
 * You MUST replace this with your actual table generated from your
 * inverse kinematics simulation. The simulation should generate hundreds
 * or thousands of entries covering your entire workspace.
 * 
 * The table is stored in PROGMEM (Flash memory) to save宝贵的 RAM on the ItsyBitsy.
 */
const IKEntry IK_TABLE[] PROGMEM = {
  //  x    y    w    {base, shoulder, elbow, wrist, grip_rot, gripper}
  {320, 240, 80,  {90, 60, 45, 30, 90, GRIPPER_OPEN}},   // Center, medium object
  {320, 240, 160, {90, 45, 60, 45, 90, GRIPPER_OPEN}},   // Center, large object (close)
  {160, 240, 80,  {60, 60, 45, 30, 90, GRIPPER_OPEN}},   // Left side
  {480, 240, 80,  {120, 60, 45, 30, 90, GRIPPER_OPEN}},  // Right side
  {320, 120, 80,  {90, 75, 30, 45, 90, GRIPPER_OPEN}},   // Top of frame
  {320, 360, 80,  {90, 45, 60, 30, 90, GRIPPER_OPEN}},   // Bottom of frame
  // ... Add MANY more entries here from your simulation ...
};
const int IK_TABLE_SIZE = sizeof(IK_TABLE) / sizeof(IKEntry);

// STATE MACHINE
// Robot operation states
enum RobotState {
  IDLE,              // Waiting for commands from Raspberry Pi
  MOVING_TO_PICK,    // Moving toward detected object (receiving POS updates)
  GRIPPING,          // Closing gripper to grab object
  MOVING_TO_BIN,     // Transporting object to appropriate bin
  RELEASING,         // Opening gripper to drop object
  RETURNING_HOME,    // Returning to home position
  ERROR_STATE        // Something went wrong
};
RobotState currentState = IDLE;  // Current state of the robot

// GLOBAL VARIABLES
// Target object data
int targetX = 0, targetY = 0;          // Current target coordinates (pixels)
int targetW = 0, targetH = 0;          // Bounding box dimensions (for depth estimation)
String targetClass = "";                // Object class (PAPER, PLASTIC, GENERAL)
BinPosition* targetBin = nullptr;       // Pointer to target bin for this object

// PID variables
int lastErrorX = 0, lastErrorY = 0;     // Previous errors for derivative calculation
int integralX = 0, integralY = 0;       // Accumulated errors for integral term
unsigned long lastPIDTime = 0;           // Last time PID was updated (for dt calculation)

// Current servo angles (for smooth interpolation)
int currentAngles[6];

// PCA9685 CONTROL FUNCTIONS

/**
 * Initialize the PCA9685 PWM driver
 * Sets up I2C communication and configures PWM frequency
 */
void initPCA9685() {
  pwm.begin();                         // Start communication with PCA9685
  pwm.setOscillatorFrequency(27000000); // Calibrate oscillator (standard for most boards)
  pwm.setPWMFreq(SERVO_FREQ);           // Set PWM frequency for servos (50Hz)
  delay(10);                            // Give time for initialization
}

/**
 * Set a specific servo to a specific angle
 * 
 * @param channel Logical servo index (0-5, not the physical channel)
 * @param angle Desired angle (0-180 degrees)
 * 
 * This function:
 *   1. Constrains angle to valid range
 *   2. Maps angle to PWM pulse width using calibrated min/max values
 *   3. Sends command to PCA9685
 */
void setServoAngle(int channel, int angle) {
  // Limit angle to 0-180 degrees
  angle = constrain(angle, 0, 180);
  
  // Map angle (0-180) to PWM pulse width (servoMin - servoMax)
  // This uses the per-channel calibration values
  uint16_t pulse = map(angle, 0, 180, servoMin[channel], servoMax[channel]);
  
  // Send PWM command to PCA9685
  // setPWM(channel, on_time, off_time) - we use 0 for on_time (start of pulse)
  pwm.setPWM(CHANNELS[channel], 0, pulse);
}

/**
 * Move all servos to specified angles immediately (no interpolation)
 * 
 * @param angles Array of 6 target angles
 */
void moveServos(const int angles[6]) {
  for (int i = 0; i < 6; i++) {
    setServoAngle(i, angles[i]);
    currentAngles[i] = angles[i];  // Update current position
  }
}

/**
 * Move all servos smoothly to target angles using linear interpolation
 * This prevents jerky movements that could destabilize the arm
 * 
 * @param targetAngles Array of 6 target angles
 * @param stepDelay Delay between interpolation steps (ms) - lower = faster movement
 */
void moveServosSmooth(const int targetAngles[6], int stepDelay = 20) {
  int startAngles[6];
  memcpy(startAngles, currentAngles, sizeof(startAngles));
  
  // Calculate how many steps each servo needs to reach target
  int maxSteps = 0;
  int steps[6];
  
  for (int i = 0; i < 6; i++) {
    steps[i] = abs(targetAngles[i] - startAngles[i]);
    if (steps[i] > maxSteps) maxSteps = steps[i];
  }
  
  if (maxSteps == 0) return;  // Already at target
  
  // Move through intermediate positions
  for (int step = 1; step <= maxSteps; step++) {
    int intermediate[6];
    for (int i = 0; i < 6; i++) {
      if (steps[i] > 0) {
        // Linear interpolation between start and target
        intermediate[i] = startAngles[i] + (targetAngles[i] - startAngles[i]) * step / maxSteps;
      } else {
        intermediate[i] = targetAngles[i];  // Servo doesn't need to move
      }
    }
    moveServos(intermediate);  // Move to intermediate position
    delay(stepDelay);           // Wait for servos to reach position
  }
}

// SERVO CALIBRATION FUNCTIONS

/**
 * Helper function to test and calibrate servo ranges
 * This moves each servo through its full range so you can:
 *   1. Verify correct wiring and operation
 *   2. Adjust SERVO_MIN and SERVO_MAX for each servo
 *   3. Identify any mechanical issues
 * 
 * Run this function once during setup to calibrate, then comment it out
 */
void calibrateServoRange() {
  Serial.println("SERVO CALIBRATION MODE");
  Serial.println("WARNING: Make sure arm has clearance to move!");
  delay(2000);
  
  for (int i = 0; i < 6; i++) {
    Serial.print("Testing Servo "); Serial.print(i);
    Serial.print(" (Channel "); Serial.print(CHANNELS[i]); Serial.println(")");
    
    // Move to minimum position
    Serial.println("  → Moving to MIN (0°)");
    setServoAngle(i, 0);
    delay(2000);
    
    // Move to center position
    Serial.println("  → Moving to CENTER (90°)");
    setServoAngle(i, 90);
    delay(2000);
    
    // Move to maximum position
    Serial.println("  → Moving to MAX (180°)");
    setServoAngle(i, 180);
    delay(2000);
    
    // Return to center
    setServoAngle(i, 90);
    delay(1000);
  }
  
  Serial.println("Calibration complete! Adjust SERVO_MIN/MAX if needed.");
  
  // Return to home position
  moveServosSmooth(HOME_ANGLES, 30);
}

// INVERSE KINEMATICS

/**
 * Look up servo angles for a given target position using pre-calculated table
 * This is MUCH faster than calculating IK in real-time on the ItsyBitsy
 * 
 * @param x Target X coordinate (pixels from camera)
 * @param y Target Y coordinate (pixels from camera)
 * @param w Bounding box width (for depth estimation)
 * @param h Bounding box height (unused but kept for API consistency)
 * @param outputAngles Array to store the 6 resulting servo angles
 * @return true if found matching position, false if no suitable entry
 */
bool ikLookup(int x, int y, int w, int h, int outputAngles[6]) {
  int bestIdx = -1;
  long minDist = 999999;
  
  // Search through entire lookup table for closest match
  for (int i = 0; i < IK_TABLE_SIZE; i++) {
    IKEntry entry;
    // Read from PROGMEM (Flash memory) into RAM
    memcpy_P(&entry, &IK_TABLE[i], sizeof(IKEntry));
    
    // Calculate weighted distance (x, y, and width/depth)
    long dx = abs(entry.x - x);
    long dy = abs(entry.y - y);
    long dw = abs(entry.w - w);
    
    // Using squared distance to avoid expensive sqrt operation
    long dist = dx*dx + dy*dy + dw*dw;
    
    if (dist < minDist) {
      minDist = dist;
      bestIdx = i;
    }
  }
  
  if (bestIdx >= 0) {
    IKEntry best;
    memcpy_P(&best, &IK_TABLE[bestIdx], sizeof(IKEntry));
    memcpy(outputAngles, best.angles, sizeof(best.angles));
    return true;
  }
  
  return false;  // No suitable position found
}

// PID CONTROL

/**
 * Update arm position based on visual feedback from Raspberry Pi
 * This implements closed-loop control:
 *   1. Calculate error between target and current position
 *   2. Compute PID output
 *   3. Convert output to angle corrections
 *   4. Apply corrections to servos
 * 
 * @param currentX Current X coordinate from camera (object position)
 * @param currentY Current Y coordinate from camera
 */
void pidUpdate(int currentX, int currentY) {
  // Only run PID when we're actively moving to pick
  if (currentState != MOVING_TO_PICK) return;
  
  // Calculate time delta since last update
  unsigned long now = micros();
  float dt = (now - lastPIDTime) / 1000000.0;  // Convert to seconds
  if (dt < 0.01) return;  // Don't update faster than 10ms (100Hz)
  lastPIDTime = now;
  
  // Calculate errors (setpoint - current value)
  int errorX = targetX - currentX;
  int errorY = targetY - currentY;
  
  // If we're close enough, no need to adjust
  if (abs(errorX) < TARGET_TOLERANCE && abs(errorY) < TARGET_TOLERANCE) {
    return;
  }
  
  // INTEGRAL TERM (accumulates over time)
  // Anti-windup: prevent integral from growing too large
  integralX += errorX * dt;
  integralY += errorY * dt;
  integralX = constrain(integralX, -100, 100);
  integralY = constrain(integralY, -100, 100);
  
  // DERIVATIVE TERM (rate of change)
  float derivativeX = (errorX - lastErrorX) / dt;
  float derivativeY = (errorY - lastErrorY) / dt;
  
  // PID OUTPUT = P + I + D
  float outputX = Kp * errorX + Ki * integralX + Kd * derivativeX;
  float outputY = Kp * errorY + Ki * integralY + Kd * derivativeY;
  
  // Convert PID output (range ~ -100 to 100) to angle corrections
  int angleCorrectionX = map(outputX, -100, 100, -5, 5);  // Base rotation correction
  int angleCorrectionY = map(outputY, -100, 100, -3, 3);  // Shoulder/elbow correction
  
  // Apply corrections to current angles
  int correctedAngles[6];
  memcpy(correctedAngles, currentAngles, sizeof(correctedAngles));
  correctedAngles[0] += angleCorrectionX;  // Adjust base
  correctedAngles[1] += angleCorrectionY;  // Adjust shoulder
  correctedAngles[2] -= angleCorrectionY;  // Compensate elbow (keeps end effector orientation)
  
  // Move to corrected position
  moveServos(correctedAngles);
  
  // Store errors for next derivative calculation
  lastErrorX = errorX;
  lastErrorY = errorY;
}

// STATE MACHINE SEQUENCES

/**
 * Start a new pick-and-place sequence
 * Called when PICK command is received from Raspberry Pi
 * 
 * @param x Target X coordinate
 * @param y Target Y coordinate
 * @param w Bounding box width (depth estimate)
 * @param h Bounding box height
 * @param className Object class (PAPER, PLASTIC, GENERAL)
 */
void startPickSequence(int x, int y, int w, int h, const String& className) {
  // Store target information
  targetX = x;
  targetY = y;
  targetW = w;
  targetH = h;
  targetClass = className;
  
  // Select appropriate bin based on object class
  if (className == "PAPER") {
    targetBin = &BIN_PAPER;
  } else if (className == "PLASTIC") {
    targetBin = &BIN_PLASTIC;
  } else {
    targetBin = &BIN_GENERAL;  // Default for GENERAL or unknown
  }
  
  // Look up pre-calculated angles for this position
  int pickAngles[6];
  if (!ikLookup(x, y, w, h, pickAngles)) {
    Serial.println("ERROR: No IK solution found for target position");
    currentState = ERROR_STATE;
    return;
  }
  
  Serial.println("Moving to pick position");
  currentState = MOVING_TO_PICK;
  moveServosSmooth(pickAngles);
  
  // Initialize PID variables
  lastErrorX = 0; lastErrorY = 0;
  integralX = 0; integralY = 0;
  lastPIDTime = micros();
}

/**
 * Called when Raspberry Pi sends TARGET_REACHED
 * This means the arm has reached the object (object no longer visible)
 * Executes the complete pick-and-place sequence:
 *   1. Close gripper to grab object
 *   2. Lift object slightly
 *   3. Move to appropriate bin
 *   4. Open gripper to release
 *   5. Return to home position
 */
void targetReached() {
  // Only valid if we were moving to pick
  if (currentState != MOVING_TO_PICK) return;
  
  //STEP 1: GRIP (close gripper)
  Serial.println("STEP 1: Gripping object");
  currentState = GRIPPING;
  
  int gripAngles[6];
  memcpy(gripAngles, currentAngles, sizeof(gripAngles));
  gripAngles[5] = GRIPPER_CLOSED;  // Close gripper
  moveServosSmooth(gripAngles, 10);  // Slow, gentle movement
  delay(500);  // Wait for gripper to fully close
  
  //STEP 2: LIFT (raise object slightly)
  Serial.println("STEP 2: Lifting object");
  
  int liftAngles[6];
  memcpy(liftAngles, currentAngles, sizeof(liftAngles));
  liftAngles[1] -= 20;  // Raise shoulder (decrease angle = move up)
  liftAngles[2] += 20;  // Compensate with elbow to keep orientation
  moveServosSmooth(liftAngles);
  
  //STEP 3: MOVE TO BIN
  Serial.println("STEP 3: Moving to bin");
  currentState = MOVING_TO_BIN;
  
  int binAngles[6];
  memcpy(binAngles, targetBin->angles, sizeof(binAngles));
  binAngles[5] = GRIPPER_CLOSED;  // Keep gripper closed during transport
  moveServosSmooth(binAngles);
  
  //STEP 4: RELEASE (open gripper)
  Serial.println("STEP 4: Releasing object into bin");
  currentState = RELEASING;
  
  int releaseAngles[6];
  memcpy(releaseAngles, currentAngles, sizeof(releaseAngles));
  releaseAngles[5] = GRIPPER_OPEN;  // Open gripper
  moveServosSmooth(releaseAngles, 10);
  delay(500);  // Wait for object to fall
  
  //STEP 5: RETURN HOME
  Serial.println("STEP 5: Returning to home position");
  currentState = RETURNING_HOME;
  moveServosSmooth(HOME_ANGLES);
  
  // Sequence complete
  currentState = IDLE;
  Serial.println("DONE");  // Notify Raspberry Pi of success
}

// SERIAL COMMUNICATION

/**
 * Process incoming commands from Raspberry Pi over USB serial
 * Expected commands:
 *   - PICK [class] [cx] [cy] [bw] [bh]  - Start picking object
 *   - POS [cx] [cy]                       - Position update for PID
 *   - TARGET_REACHED                       - Object reached, start gripping
 */
void processSerialCommand() {
  if (!Serial.available()) return;
  
  // Read command until newline
  String command = Serial.readStringUntil('\n');
  command.trim();  // Remove whitespace
  
  if (command.length() == 0) return;
  
  // Extract command word (first part before space)
  int firstSpace = command.indexOf(' ');
  String cmd = (firstSpace > 0) ? command.substring(0, firstSpace) : command;
  cmd.toUpperCase();
  
  //PICK COMMAND
  if (cmd == "PICK") {
    // Format: PICK PAPER 320 240 80 60
    if (firstSpace <= 0) {
      Serial.println("ERROR: Invalid PICK format");
      return;
    }
    
    // Parse class and coordinates
    String rest = command.substring(firstSpace + 1);
    int idx1 = rest.indexOf(' ');
    String classStr = rest.substring(0, idx1);
    
    int cx, cy, bw, bh;
    if (sscanf(rest.substring(idx1 + 1).c_str(), "%d %d %d %d", &cx, &cy, &bw, &bh) != 4) {
      Serial.println("ERROR: Invalid coordinate format");
      return;
    }
    
    // Start the pick sequence
    startPickSequence(cx, cy, bw, bh, classStr);
  }
  
  //POS COMMAND (position update)
  else if (cmd == "POS") {
    // Format: POS 318 242
    if (currentState != MOVING_TO_PICK) {
      // Ignore POS commands if not in correct state
      return;
    }
    
    int cx, cy;
    if (sscanf(command.c_str(), "POS %d %d", &cx, &cy) == 2) {
      pidUpdate(cx, cy);
    }
  }
  
  //TARGET_REACHED COMMAND
  else if (cmd == "TARGET_REACHED") {
    targetReached();
  }
  
  //UNKNOWN COMMAND
  else {
    Serial.println("ERROR: Unknown command");
  }
}

// SETUP AND MAIN LOOP

/**
 * Arduino setup function - runs once at startup
 */
void setup() {
  // Initialize serial communication (match baud rate with Raspberry Pi)
  Serial.begin(9600);
  while (!Serial) delay(10);  // Wait for serial connection
  
  Serial.println("ItsyBitsy M4 + PCA9685 Robot Arm Controller");
  Serial.println("Ready for commands from Raspberry Pi");
  
  // Initialize I2C and PCA9685
  Wire.begin();           // Start I2C communication
  initPCA9685();          // Configure PWM driver
  
  // Move to home position at startup
  Serial.println("Moving to home position...");
  moveServosSmooth(HOME_ANGLES, 30);
  
  // OPTIONAL: Uncomment to run calibration (run once, then comment out)
  // calibrateServoRange();
  
  currentState = IDLE;
  Serial.println("System ready! Waiting for commands...");
}

/**
 * Arduino main loop - runs continuously
 */
void loop() {
  // Check for and process incoming commands
  processSerialCommand();
  
  // SAFETY TIMEOUT: If we're moving to pick but stop receiving POS updates,
  // go to error state and return home to prevent damage
  static unsigned long lastPosTime = micros();
  if (currentState == MOVING_TO_PICK) {
    if (micros() - lastPosTime > 5000000) {  // 5 second timeout
      Serial.println("ERROR: Position update timeout - no POS received");
      currentState = ERROR_STATE;
      moveServosSmooth(HOME_ANGLES);  // Return to safe position
      currentState = IDLE;
    }
  } else {
    lastPosTime = micros();  // Reset timer when not in MOVING_TO_PICK
  }
  
  // Small delay to prevent overwhelming the CPU
  delay(1);
}