import csv

import numpy as np
from matplotlib import pyplot as plt
from numpy import array
from skimage.measure import LineModelND, ransac

from ransac_circle import fit_circle_ransac
from ransac_triangle import fit_triangle_ransac
from ransac_square import fit_square_ransac
from ransac_pentagon import fit_pentagon_ransac
from ransac_rhombus import fit_rhombus_ransac
from draw_figures import draw_circle, draw_polygon

def read_coordinates_from_csv(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                x, y = map(float, row[:2])
                coordinates.append([x, y])
    return np.array(coordinates)

def find_possible_poses_from_csv(coordinates: np.ndarray) -> ([(float, float)], [float], [float]):
    data = np.array(coordinates)
    current_data = data

    inliers_list = []
    outliers_list = []
    models_list = []

    for _ in range(4):
        model, inliers_bools = ransac(current_data, LineModelND, min_samples=2,
                                      residual_threshold=0.007, max_trials=10000)

        inliers = np.array([point for (point, inlier_bool) in zip(current_data, inliers_bools) if inlier_bool])
        outliers = np.array([point for (point, inlier_bool) in zip(current_data, inliers_bools) if not inlier_bool])

        inliers_list.append(inliers)
        outliers_list.append(outliers)
        models_list.append(model)
        current_data = outliers

    draw_walls(data, inliers_list[0], inliers_list[1],
               inliers_list[2], inliers_list[3])

def draw_walls(data: array, inliers1: array, inliers2: array, inliers3: array, inliers4: array) -> None:
    data_x, data_y = zip(*data)

    inliers1_x, inliers1_y = zip(*inliers1)
    inliers2_x, inliers2_y = zip(*inliers2)
    inliers3_x, inliers3_y = zip(*inliers3)
    inliers4_x, inliers4_y = zip(*inliers4)

    plt.scatter(inliers1_x, inliers1_y, color='blue', marker='x')
    plt.scatter(inliers2_x, inliers2_y, color='blue', marker='x')
    plt.scatter(inliers3_x, inliers3_y, color='blue', marker='x')
    plt.scatter(inliers4_x, inliers4_y, color='blue', marker='x')

    plt.gca().set_aspect('equal')
    plt.xlim(min(*data_x, 0) - 0.1, max(*data_x, 0) + 0.1)
    plt.ylim(min(*data_y, 0) - 0.1, max(*data_y, 0) + 0.1)

def show_plot(outliers):
    if len(outliers) > 0:
        outliers_x, outliers_y = zip(*outliers)
        plt.scatter(outliers_x, outliers_y, color='red', label='Outliers', marker='x')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mapping of the primitives using RANSAC')

    plt.grid(True)
    plt.show()

def write_shape_details(file, shape_type, centroid, vertices):
    file.write(f'{shape_type} = {centroid}, {vertices}\n')

def fit_and_draw_shape(shape_type, fit_function, outliers, file, vertex_count):
    best_shape, inliers, outliers = fit_function(outliers)

    if best_shape is not None:
        draw_polygon(best_shape)
        print(f'{shape_type} found: {best_shape}')
        centroid = [sum(x) / vertex_count for x in zip(*best_shape)]
        write_shape_details(file, shape_type[0], centroid, best_shape)
    else:
        print(f"No {shape_type} found")

    return outliers

def main() -> None:
    coordinates = read_coordinates_from_csv("random_map_points.csv")
    show_plot(coordinates)

    find_possible_poses_from_csv(coordinates[:2796])
    outliers = coordinates[2796:]

    with open('shapes.txt', 'w') as file:
        pass

    with open('shapes.txt', 'w') as file:
        print("Number of outliers:", len(outliers))

        best_circle, inliers, outliers = fit_circle_ransac(outliers)
        if best_circle is not None:
            center = best_circle[:2]
            radius = best_circle[2]
            draw_circle(center, radius)
            file.write(f'circle found: center={center}, radius={radius}\n')
            print(f"circle found: {center}")
        else:
            print("No circle found")

        outliers = fit_and_draw_shape('triangle', fit_triangle_ransac, outliers, file, 3)
        outliers = fit_and_draw_shape('square', fit_square_ransac, outliers, file, 4)
        outliers = fit_and_draw_shape('pentagon', fit_pentagon_ransac, outliers, file, 5)
        outliers = fit_and_draw_shape('rhombus', fit_rhombus_ransac, outliers, file, 4)

    show_plot(outliers)

if __name__ == '__main__':
    main()