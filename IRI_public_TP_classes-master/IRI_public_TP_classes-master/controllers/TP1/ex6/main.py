"""
IRI - TP1 - Ex 6
By: Gonçalo Leão
"""

from controller import Robot
from controllers.utils import cmd_vel

robot: Robot = Robot()

cur_angular_vel: float = 0
while True:
    cmd_vel(robot, 0.1, cur_angular_vel)
    robot.step()
    cur_angular_vel += 0.003
