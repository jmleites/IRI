"""
IRI - TP1 - Ex 8
By: Gonçalo Leão
"""
import math

from controller import Robot, Lidar, LidarPoint


def point_to_string(point: LidarPoint) -> str:
    return "(" + str(point.x) + ", " + str(point.y) + ")"


def main():
    robot: Robot = Robot()

    timestep: int = int(robot.getBasicTimeStep())  # in ms

    lidar: Lidar = robot.getDevice('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()
    robot.step()
    for (measured_dist, point) in zip(lidar.getRangeImage(), lidar.getPointCloud()):
        # Sanity check: here I am confirming that the measured distance from the range image ...
        # ... is consistent with the XYZ coords of the point cloud
        computed_dist: float = math.hypot(point.x, point.y, point.z)
        angle: float = math.atan2(point.y, point.x)  # angle between the direction the robot is facing and the ray
        print("(" + str(point.x) + ", " + str(point.y) + " - " + str(measured_dist) + " - " + str(computed_dist) + " - " + str(angle) + ")")
    while robot.step() != -1:
        pass


if __name__ == '__main__':
    main()
