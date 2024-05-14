import csv

import numpy as np
from matplotlib import pyplot as plt
from numpy import array
from skimage.measure import LineModelND, ransac

from ransac_circle import fit_circle_ransac
from ransac_triangle import fit_triangle_ransac
from draw_figures import draw_circle, draw_triangle

def read_coordinates_from_csv(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                x, y = map(float, row[:2])  # Assuming first two columns are x, y coordinates
                coordinates.append([x, y])
    return np.array(coordinates)

def find_possible_poses_from_csv(coordinates: np.ndarray) -> ([(float, float)], [float], [float]):
    # Structure the coordinates for input to RANSAC
    data = array(coordinates)

    # Initialize lists to store inliers, outliers, and models
    inliers_list = []
    outliers_list = []
    models_list = []

    # Find the first line
    model1, inliers_bools1 = ransac(data, LineModelND, min_samples=2,
                                    residual_threshold=0.007, max_trials=10000)
    # Retrieve the outliers
    outliers1: array = array([point for (point, inlier_bool) in zip(data, inliers_bools1) if not inlier_bool])

    # Find the second line
    assert len(outliers1) >= 2, "Cannot detect the second wall!!"
    model2, inliers_bools2 = ransac(outliers1, LineModelND, min_samples=2,
                                    residual_threshold=0.007, max_trials=10000)
    # Retrieve the outliers
    outliers2: array = array([point for (point, inlier_bool) in zip(outliers1, inliers_bools2) if not inlier_bool])

    # Find the third line
    model3, inliers_bools3 = ransac(outliers2, LineModelND, min_samples=2,
                                    residual_threshold=0.007, max_trials=10000)
    # Retrieve the outliers
    outliers3: array = array([point for (point, inlier_bool) in zip(outliers2, inliers_bools3) if not inlier_bool])

    # Find the fourth line
    model4, inliers_bools4 = ransac(outliers3, LineModelND, min_samples=2,
                                    residual_threshold=0.007, max_trials=10000)
    # Retrieve the outliers
    outliers4: array = array([point for (point, inlier_bool) in zip(outliers3, inliers_bools4) if not inlier_bool])

    # Compute the inliers
    inliers1: array = array([point for (point, inlier_bool) in zip(data, inliers_bools1) if inlier_bool])
    inliers2: array = array([point for (point, inlier_bool) in zip(outliers1, inliers_bools2) if inlier_bool])
    inliers3: array = array([point for (point, inlier_bool) in zip(outliers2, inliers_bools3) if inlier_bool])
    inliers4: array = array([point for (point, inlier_bool) in zip(outliers3, inliers_bools4) if inlier_bool])

    # Add inliers and outliers to lists
    inliers_list.extend([inliers1, inliers2, inliers3, inliers4])
    outliers_list.extend([outliers1, outliers2, outliers3, outliers4])
    models_list.extend([model1, model2, model3, model4])

    draw_walls(data, model1, inliers1, model2, inliers2, model3, inliers3, model4, inliers4, outliers4)

    # Return only the outliers from the final iteration
    return inliers_list[-1], outliers_list[-1], models_list[-1]

def draw_walls(data: array, line1: LineModelND, inliers1: array, line2: LineModelND, inliers2: array, line3: LineModelND, inliers3: array, line4: LineModelND, inliers4: array,
               outliers: array) -> None:
    # Unpack the points into separate lists of x and y coordinates
    data_x, data_y = zip(*data)
    inliers1_x, inliers1_y = zip(*inliers1)
    inliers2_x, inliers2_y = zip(*inliers2)
    inliers3_x, inliers3_y = zip(*inliers3)
    inliers4_x, inliers4_y = zip(*inliers4)
    outliers_x = []
    outliers_y = []
    if len(outliers) > 0:
        outliers_x, outliers_y = zip(*outliers)

    # Plot the data points with different colors
    plt.scatter(inliers1_x, inliers1_y, color='blue', label='Inliers1', marker='x')
    plt.scatter(inliers2_x, inliers2_y, color='blue', label='Inliers2', marker='x')
    plt.scatter(inliers3_x, inliers3_y, color='blue', label='Inliers3', marker='x')
    plt.scatter(inliers4_x, inliers4_y, color='blue', label='Inliers4', marker='x')

    # Add the computed lines to the plot
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


def show_plot(outliers):
    if len(outliers) > 0:
        outliers_x, outliers_y = zip(*outliers)
        plt.scatter(outliers_x, outliers_y, color='red', label='Outliers', marker='x')
    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('LiDAR points and lines relative to the robot')
    # Show the plot
    plt.grid(True)
    plt.show()

def get_model_line_points(points_x_coords: [float], points_y_coords: [float], origin: (float, float),
                          direction: (float, float)) -> (np.ndarray, np.ndarray):
    if direction[0] == 0:  # vertical line
        line_min_y: float = min(points_y_coords)
        line_max_y: float = max(points_y_coords)
        line_y_values: np.ndarray[np.dtype:float] = np.linspace(line_min_y - 0.1, line_max_y + 0.1, 100)

        line_x_values: np.ndarray[np.dtype:float] = np.full(shape=len(line_y_values), fill_value=points_x_coords[0],
                                                            dtype=np.float64)
    else:
        line_min_x: float = min(points_x_coords)
        line_max_x: float = max(points_x_coords)
        line_x_values: np.ndarray[np.dtype:float] = np.linspace(line_min_x - 0.1, line_max_x + 0.1, 100)

        slope: float = direction[1] / direction[0]
        intercept: float = origin[1] - slope * origin[0]
        line_y_values: np.ndarray[np.dtype:float] = slope * line_x_values + intercept
    return line_x_values, line_y_values

def main() -> None:
    csv_file_path = r'C:\uni\Intr. Robótica\IRI_public_TP_classes-master\IRI_public_TP_classes-master\worlds\custom_maps\triangle_points.csv'
    coordinates = read_coordinates_from_csv(csv_file_path)

    find_possible_poses_from_csv(coordinates)

    inliers, outliers, models = find_possible_poses_from_csv(coordinates)

    if len(outliers) > 3:
        best_circle, inliers, outliers = fit_circle_ransac(outliers)
        if best_circle is not None:
            center = best_circle[:2]
            radius = best_circle[2]
            print(center, radius)
            draw_circle(center, radius)

    if len(outliers) > 3:
        best_triangle, inliers, outliers = fit_triangle_ransac(outliers)
        if best_triangle is not None:
            draw_triangle(best_triangle)

    show_plot(outliers)

if __name__ == '__main__':
    main()