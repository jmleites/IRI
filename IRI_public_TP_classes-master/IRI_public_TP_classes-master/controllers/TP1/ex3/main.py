"""
IRI - TP1 - Ex 3
By: Gonçalo Leão
"""

import math
from controller import Robot
from controllers.utils import cmd_vel

robot: Robot = Robot()

print('forward')
cmd_vel(robot, 0.1, 0)
robot.step(1000)

print('stop')
cmd_vel(robot, 0, 0)
robot.step(1000)

print('spin')
cmd_vel(robot, 0, -math.pi)
while robot.step() != -1:
    pass
