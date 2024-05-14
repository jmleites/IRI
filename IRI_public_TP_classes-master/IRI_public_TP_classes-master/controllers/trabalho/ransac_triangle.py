import numpy as np
import copy


def fit_triangle_ransac(points, max_iter=2000, threshold=0.01, min_inliers=300):
    best_triangle = None
    best_inliers = []
    best_outliers = copy.copy(points)
    num_points = len(points)

    for _ in range(max_iter):
        sample_indices = np.random.choice(num_points, 3, replace=False)
        sampled_points = points[sample_indices]

        triangle = sampled_points
        vertices = sampled_points[:6].reshape((3, 2))

        distances = np.array([point_to_triangle_distance(point, vertices) for point in points])
        outliers = points[distances > threshold]
        inliers = points[distances <= threshold]

        if len(inliers) >= min_inliers:
            if len(inliers) > len(best_inliers):
                best_triangle = triangle
                best_inliers = inliers
                best_outliers = outliers

    return best_triangle, best_inliers, best_outliers

def point_to_triangle_distance(point, vertices):
    edge_distances = []
    for i in range(3):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 3]
        edge_distances.append(point_to_line_distance(point, p1, p2))

    return np.min(edge_distances)


def point_to_line_distance(point, p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    num = np.abs(y_diff * point[0] - x_diff * point[1] + p2[0] * p1[1] - p2[1] * p1[0])
    denom = np.sqrt(y_diff ** 2 + x_diff ** 2)
    return num / denom
