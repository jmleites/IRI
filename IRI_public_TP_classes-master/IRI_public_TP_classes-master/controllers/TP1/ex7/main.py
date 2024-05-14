"""
IRI - TP1 - Ex 7
By: Gonçalo Leão
"""

from controller import Robot
from controllers.utils import cmd_vel

robot: Robot = Robot()

linear_vel: float = 0.1
max_radius: float = 0.25
min_angular_vel: float = linear_vel / max_radius

cur_angular_vel: float = 5
while True:
    cmd_vel(robot, 0.1, cur_angular_vel)
    robot.step()
    cur_angular_vel = max(min_angular_vel, cur_angular_vel - 0.005)
