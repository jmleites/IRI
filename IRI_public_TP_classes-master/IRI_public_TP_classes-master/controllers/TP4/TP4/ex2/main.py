"""
IRI - TP4 - Ex 2
By: Gonçalo Leão
"""
import math
from typing import Union, Tuple

from matplotlib import pyplot as plt

from controller import Robot, Lidar, LidarPoint, Compass, GPS, Keyboard
import numpy as np

from controllers.TP4.occupancy_grid import OccupancyGrid
from controllers.transformations import create_tf_matrix, get_translation
from controllers.utils import cmd_vel, bresenham_extended


class ProbabilisticOccupancyGrid(OccupancyGrid):
    def __init__(self, origin: (float, float), dimensions: (int, int), resolution: float):
        super().__init__(origin, dimensions, resolution)

        # Initialize the grid
        self.log_prior = self.get_log_odd(0.5)
        self.occupancy_grid: np.ndarray = np.full(dimensions, self.log_prior, dtype=np.float32)

    def update_map(self, robot_tf: np.ndarray, lidar_points: [LidarPoint]) -> None:
        # Get the grid coord for the robot pose
        robot_coord: (int, int) = self.real_to_grid_coords(get_translation(robot_tf)[0:2])

        # Get the grid coords and distances for the lidar points
        grid_lidar_coords: [(int, int)] = []
        measured_dists: [float] = []
        for point in lidar_points:
            coord: (int, int) = self.real_to_grid_coords(np.dot(robot_tf, [point.x, point.y, 0.0, 1.0])[0:2])
            grid_lidar_coords.append(coord)
            measured_dists.append(math.hypot(point.x, point.y))

        # Update the cell of the robot's position
        self.update_cell(robot_coord, True, 0)

        # Update the cells on the lines defined by the lidar points
        for (coord, measured_dist) in zip(grid_lidar_coords, measured_dists):
            for line_coord in bresenham_extended(robot_coord, coord, (0, 0), (self.dimensions[0]-1, self.dimensions[1]-1))[1:]:
                cell_dist: float = self.resolution*math.hypot(line_coord[0] - robot_coord[0], line_coord[1] - robot_coord[1])
                self.update_cell(line_coord, False, cell_dist - measured_dist)

    def update_cell(self, coords: (int, int), is_robot_cell: bool, cell_minus_measured_distance_to_robot: float) -> None:
        if self.are_grid_coords_in_bounds(coords):
            # Update the grid cell
            inverse_sensor_model_prob: float = 0.1
            if not is_robot_cell:
                inverse_sensor_model_prob = np.interp(cell_minus_measured_distance_to_robot, [-0.03, 0, 0.03], [0.3, 0.9, 0.5], left=0.3, right=0.5)
            self.occupancy_grid[coords] = self.occupancy_grid[coords] + self.get_log_odd(inverse_sensor_model_prob) - self.log_prior

    def get_probability_grid(self) -> np.ndarray:
        def my_vectorized_func(m):
            return np.round(2*(1 - (1 / (1 + np.exp(m)))))/2

        return my_vectorized_func(self.occupancy_grid)

    def get_log_odd(self, prob: float) -> float:
        return math.log(prob/(1 - prob))


def main() -> None:
    robot: Robot = Robot()
    timestep: int = 100  # in ms

    kb: Keyboard = Keyboard()
    kb.enable(timestep)

    keyboard_linear_vel: float = 0.3
    keyboard_angular_vel: float = 1.5

    map: ProbabilisticOccupancyGrid = ProbabilisticOccupancyGrid([0.0, 0.0], [200, 200], 0.01)

    lidar: Lidar = robot.getDevice('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()

    compass: Compass = robot.getDevice('compass')
    compass.enable(timestep)

    gps: GPS = robot.getDevice('gps')
    gps.enable(timestep)

    scan_count: int = 0
    while robot.step(timestep) != -1:
        key: int = kb.getKey()
        if key == ord('W'):
            cmd_vel(robot, keyboard_linear_vel, 0)
        elif key == ord('S'):
            cmd_vel(robot, -keyboard_linear_vel, 0)
        elif key == ord('A'):
            cmd_vel(robot, 0, keyboard_angular_vel)
        elif key == ord('D'):
            cmd_vel(robot, 0, -keyboard_angular_vel)
        else:  # Not a movement key
            cmd_vel(robot, 0, 0)
            if key == ord(' '):
                scan_count += 1
                print('scan count: ', scan_count)

                # Read the robot's pose
                gps_readings: [float] = gps.getValues()
                robot_position: (float, float) = (gps_readings[0], gps_readings[1])
                compass_readings: [float] = compass.getValues()
                robot_orientation: float = math.atan2(compass_readings[0], compass_readings[1])
                robot_tf: np.ndarray = create_tf_matrix((robot_position[0], robot_position[1], 0.0), robot_orientation)

                # Read the LiDAR and update the map
                map.update_map(robot_tf, lidar.getPointCloud())

                # Show the updated map
                plt.imshow(np.flip(map.get_probability_grid(), 0))
                plt.savefig('ex2-scan' + str(scan_count) + '.png')
                # plt.show()


if __name__ == '__main__':
    main()
