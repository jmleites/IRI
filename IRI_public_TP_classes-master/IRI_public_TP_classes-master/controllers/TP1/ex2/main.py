"""
IRI - TP1 - Ex 2
By: Gonçalo Leão
"""

from controller import Robot
from controllers.utils import cmd_vel

robot: Robot = Robot()

# linear_vel = radius * angular_vel
radius: float = 0.125
linear_vel: float = 0.1
angular_vel: float = linear_vel / radius
cmd_vel(robot, linear_vel, angular_vel)

# Main loop
while robot.step() != -1:
    pass
