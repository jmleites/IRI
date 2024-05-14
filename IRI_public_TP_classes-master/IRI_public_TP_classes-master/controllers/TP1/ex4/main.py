"""
IRI - TP1 - Ex 4
By: Gonçalo Leão
"""

from controller import Robot
from controllers.utils import cmd_vel, move_forward

robot: Robot = Robot()

while True:
    print('forward')
    move_forward(robot, 0.25, 0.1)

    print('stop')
    cmd_vel(robot, 0, 0)
    robot.step(1000)

    print('backward')
    move_forward(robot, 0.25, -0.1)

    print('stop')
    cmd_vel(robot, 0, 0)
    robot.step(1000)
