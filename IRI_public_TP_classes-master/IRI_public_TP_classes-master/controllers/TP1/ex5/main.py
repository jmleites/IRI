"""
IRI - TP1 - Ex 5
By: Gonçalo Leão
"""

import math
from controller import Robot
from controllers.utils import move_forward, rotate

robot: Robot = Robot()

while True:
    print('forward')
    move_forward(robot, 0.25, 0.1)

    print('rotate')
    rotate(robot, math.pi / 2, 1)

