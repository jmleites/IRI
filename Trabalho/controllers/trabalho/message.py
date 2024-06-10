import math
import numpy as np

from numpy import array
from skimage.measure import LineModelND, ransac
from ransac_circle import fit_circle_ransac

from matplotlib import pyplot as plt
from controller import Robot, Lidar, LidarPoint, Compass, GPS, Keyboard

from controllers.TP4.occupancy_grid import OccupancyGrid
from controllers.transformations import create_tf_matrix, get_translation
from controllers.utils import cmd_vel, bresenham_extended

class ProbabilisticOccupancyGrid(OccupancyGrid):
    def __init__(self, origin: (float, float), dimensions: (int, int), resolution: float):
        super().__init__(origin, dimensions, resolution)
        self.log_prior = self.get_log_odd(0.5)
        self.occupancy_grid: np.ndarray = np.full(dimensions, self.log_prior, dtype=np.float32)

    def get_occupancy_grid(self) -> np.ndarray:
        return self.occupancy_grid

    def update_map(self, robot_tf: np.ndarray, lidar_points: [LidarPoint]) -> None:
        robot_coord: (int, int) = self.real_to_grid_coords(get_translation(robot_tf)[0:2])
        grid_lidar_coords: [(int, int)] = []
        measured_dists: [float] = []
        for point in lidar_points:
            coord: (int, int) = self.real_to_grid_coords(np.dot(robot_tf, [point.x, point.y, 0.0, 1.0])[0:2])
            grid_lidar_coords.append(coord)
            measured_dists.append(math.hypot(point.x, point.y))
        self.update_cell(robot_coord, True, 0)
        for (coord, measured_dist) in zip(grid_lidar_coords, measured_dists):
            for line_coord in bresenham_extended(robot_coord, coord, (0, 0), (self.dimensions[0]-1, self.dimensions[1]-1))[1:]:
                cell_dist: float = self.resolution * math.hypot(line_coord[0] - robot_coord[0], line_coord[1] - robot_coord[1])
                self.update_cell(line_coord, False, cell_dist - measured_dist)

    def update_cell(self, coords: (int, int), is_robot_cell: bool, cell_minus_measured_distance_to_robot: float) -> None:
        if self.are_grid_coords_in_bounds(coords):
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

def read_coordinates_from_grid(grid):
    coordinates = []
    for row in grid:
        if len(row) >= 2:
            x, y = map(float, row[:2])  # Assuming first two elements are x, y coordinates
            coordinates.append([x, y])
    return np.array(coordinates)

def detect_primitives_ransac(coordinates: np.ndarray) -> ([(float, float)], [float], [float]):
    data = array(coordinates)
    inliers_list = []
    outliers_list = []
    models_list = []
    model1, inliers_bools1 = ransac(data, LineModelND, min_samples=2,
                                    residual_threshold=0.007, max_trials=10000)
    outliers1: array = array([point for (point, inlier_bool) in zip(data, inliers_bools1) if not inlier_bool])
    assert len(outliers1) >= 2, "Cannot detect the second wall!!"
    model2, inliers_bools2 = ransac(outliers1, LineModelND, min_samples=2,
                                    residual_threshold=0.007, max_trials=10000)
    outliers2: array = array([point for (point, inlier_bool) in zip(outliers1, inliers_bools2) if not inlier_bool])
    assert len(outliers2) >= 2, "Cannot detect the third wall!!"
    model3, inliers_bools3 = ransac(outliers2, LineModelND, min_samples=2,
                                    residual_threshold=0.007, max_trials=10000)
    outliers3: array = array([point for (point, inlier_bool) in zip(outliers2, inliers_bools3) if not inlier_bool])
    assert len(outliers3) >= 2, "Cannot detect the fourth wall!!"
    model4, inliers_bools4 = ransac(outliers3, LineModelND, min_samples=2,
                                    residual_threshold=0.007, max_trials=10000)
    outliers4: array = array([point for (point, inlier_bool) in zip(outliers3, inliers_bools4) if not inlier_bool])
    inliers1: array = array([point for (point, inlier_bool) in zip(data, inliers_bools1) if inlier_bool])
    inliers2: array = array([point for (point, inlier_bool) in zip(outliers1, inliers_bools2) if inlier_bool])
    inliers3: array = array([point for (point, inlier_bool) in zip(outliers2, inliers_bools3) if inlier_bool])
    inliers4: array = array([point for (point, inlier_bool) in zip(outliers3, inliers_bools4) if inlier_bool])
    inliers_list.extend([inliers1, inliers2, inliers3, inliers4])
    outliers_list.extend([outliers1, outliers2, outliers3, outliers4])
    models_list.extend([model1, model2, model3, model4])
    draw_walls(data, model1, inliers1, model2, inliers2, model3, inliers3, model4, inliers4, outliers4)
    return inliers_list[-1], outliers_list[-1], models_list[-1]

def draw_walls(data: array, line1: LineModelND, inliers1: array, line2: LineModelND, inliers2: array, line3: LineModelND, inliers3: array, line4: LineModelND, inliers4: array,
               outliers: array) -> None:
    data_x, data_y = zip(*data)
    inliers1_x, inliers1_y = zip(*inliers1)
    inliers2_x, inliers2_y = zip(*inliers2)
    inliers3_x, inliers3_y = zip(*inliers3)
    inliers4_x, inliers4_y = zip(*inliers4)
    outliers_x = []
    outliers_y = []
    if len(outliers) > 0:
        outliers_x, outliers_y = zip(*outliers)
    plt.scatter(inliers1_x, inliers1_y, color='blue', label='Inliers1', marker='x')
    plt.scatter(inliers2_x, inliers2_y, color='blue', label='Inliers2', marker='x')
    plt.scatter(inliers3_x, inliers3_y, color='blue', label='Inliers3', marker='x')
    plt.scatter(inliers4_x, inliers4_y, color='blue', label='Inliers4', marker='x')
    line1_x_values, line1_y_values = get_model_line_points(inliers1_x, inliers1_y, line1.params[0], line1.params[1])
    plt.plot(line1_x_values, line1_y_values, color='blue', label='Line1')
    line2_x_values, line2_y_values = get_model_line_points(inliers2_x, inliers2_y, line2.params[0], line2.params[1])
    plt.plot(line2_x_values, line2_y_values, color='blue', label='Line2')
    line3_x_values, line3_y_values = get_model_line_points(inliers3_x, inliers3_y, line3.params[0], line3.params[1])
    plt.plot(line3_x_values, line3_y_values, color='blue', label='Line3')
    line4_x_values, line4_y_values = get_model_line_points(inliers4_x, inliers4_y, line4.params[0], line4.params[1])
    plt.plot(line4_x_values, line4_y_values, color='blue', label='Line4')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().set_aspect('equal')
    plt.xlim(min(*data_x, 0) - 0.1, max(*data_x, 0) + 0.1)
    plt.ylim(min(*data_y, 0) - 0.1, max(*data_y, 0) + 0.1)

def draw_circle(circle_center: tuple, circle_radius: float) -> None:
    angles = np.linspace(0, 2 * np.pi, 100)
    circle_x = circle_center[0] + circle_radius * np.cos(angles)
    circle_y = circle_center[1] + circle_radius * np.sin(angles)
    plt.plot(circle_x, circle_y, color='blue', label='Circle', linewidth=3)
    plt.scatter(circle_center[0], circle_center[1], color='blue', label='Center', marker='o')

def show_plot(outliers):
    if len(outliers) > 0:
        outliers_x, outliers_y = zip(*outliers)
        plt.scatter(outliers_x, outliers_y, color='red', label='Outliers', marker='x')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('LiDAR points and lines relative to the robot')
    plt.grid(True)
    plt.show()

def get_model_line_points(points_x_coords: [float], points_y_coords: [float], origin: (float, float), direction: (float, float)) -> (np.ndarray, np.ndarray):
    if direction[0] == 0:  # vertical line
        line_min_y: float = min(points_y_coords)
        line_max_y: float = max(points_y_coords)
        line_y_values: np.ndarray[np.dtype:float] = np.linspace(line_min_y - 0.1, line_max_y + 0.1, 100)
        line_x_values: np.ndarray[np.dtype:float] = np.full(shape=len(line_y_values), fill_value=points_x_coords[0], dtype=np.float64)
    else:
        line_min_x: float = min(points_x_coords)
        line_max_x: float = max(points_x_coords)
        line_x_values: np.ndarray[np.dtype:float] = np.linspace(line_min_x - 0.1, line_max_x + 0.1, 100)
        slope: float = direction[1] / direction[0]
        intercept: float = origin[1] - slope * origin[0]
        line_y_values: np.ndarray[np.dtype:float] = slope * line_x_values + intercept
    return line_x_values, line_y_values

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
                gps_readings: [float] = gps.getValues()
                robot_position: (float, float) = (gps_readings[0], gps_readings[1])
                compass_readings: [float] = compass.getValues()
                robot_orientation: float = math.atan2(compass_readings[0], compass_readings[1])
                robot_tf: np.ndarray = create_tf_matrix((robot_position[0], robot_position[1], 0.0), robot_orientation)
                map.update_map(robot_tf, lidar.getPointCloud())
                plt.imshow(np.flip(map.get_probability_grid(), 0))
                plt.savefig('ex2-scan' + str(scan_count) + '.png')
            if key == ord('T'):
                occupancy_grid = map.get_occupancy_grid()
                coordinates = read_coordinates_from_grid(occupancy_grid)
                print(coordinates)
                inliers, outliers, models = detect_primitives_ransac(coordinates)
                x_data = [point[0] for point in outliers]
                y_data = [point[1] for point in outliers]
                if len(outliers) > 3:
                    best_circle, inliers, outliers = fit_circle_ransac(outliers)
                    if best_circle is not None:
                        center = best_circle[:2]
                        radius = best_circle[2]
                        draw_circle(center, radius)
                show_plot(outliers)

if __name__ == "__main__":
    main()
