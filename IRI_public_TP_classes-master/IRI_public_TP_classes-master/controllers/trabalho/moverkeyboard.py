from controller import Robot, Keyboard, Lidar

# Initialize the robot, keyboard, and LiDAR sensor
robot = Robot()
keyboard = Keyboard()
keyboard.enable(100)  # Enable keyboard with sampling period of 100ms
lidar = robot.getLidar('lidar')
lidar.enable(100)  # Enable LiDAR with sampling period of 100ms

# Main loop
while robot.step(100) != -1:
    # Read keyboard input
    key = keyboard.getKey()

    # Move robot based on keyboard input
    if key == ord('W') or key == ord('w'):  # Move forward
        robot.setSpeed(1, 1)
    elif key == ord('S') or key == ord('s'):  # Move backward
        robot.setSpeed(-1, -1)
    elif key == ord('A') or key == ord('a'):  # Turn left
        robot.setSpeed(-1, 1)
    elif key == ord('D') or key == ord('d'):  # Turn right
        robot.setSpeed(1, -1)
    else:
        robot.setSpeed(0, 0)  # Stop


