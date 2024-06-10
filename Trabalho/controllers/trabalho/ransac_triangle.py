import numpy as np
import copy

def fit_triangle_ransac(points, max_iter=100000, threshold=0.01, min_inliers=200):
    best_triangle = None
    best_inliers = []
    best_outliers = copy.copy(points)
    num_points = len(points)

    for _ in range(max_iter):
        sample_indices = np.random.choice(num_points, 3, replace=False)
        sampled_points = points[sample_indices]

        if not check_angles(sampled_points):
            continue

        triangle = sampled_points
        vertices = sampled_points[:6].reshape((3, 2))

        distances = np.array([point_to_triangle_distance(point, vertices) for point in points])
        outliers = points[distances > threshold]
        inliers = points[distances <= threshold]

        if outliers is None:
            return triangle, best_inliers, best_outliers

        if len(inliers) >= min_inliers:
            if len(inliers) > len(best_inliers):
                best_triangle = triangle
                best_inliers = inliers
                best_outliers = outliers

    return best_triangle, best_inliers, best_outliers

def check_angles(vertices):
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        angle_rad = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[1]
    v3 = vertices[0] - vertices[2]

    angles = [
        angle_between_vectors(v1, -v3),
        angle_between_vectors(v2, -v1),
        angle_between_vectors(v3, -v2)
    ]

    if not are_points_ccw(vertices):
        return False

    for angle in angles:
        if not (58 <= angle <= 62):
            return False
    return True

def are_points_ccw(vertices):
    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[1]
    v3 = vertices[0] - vertices[2]

    cross_products = [
        cross_product(v1, -v3),
        cross_product(v2, -v1),
        cross_product(v3, -v2)
    ]

    return np.all(np.sign(cross_products) == np.sign(cross_products[0])) or np.all(cross_products == 0)

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
