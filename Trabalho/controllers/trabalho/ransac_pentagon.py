import numpy as np
import copy

def fit_pentagon_ransac(points, max_iter=100000, threshold=0.01, min_inliers=250):
    best_pentagon = None
    best_inliers = []
    best_outliers = copy.copy(points)
    num_points = len(points)

    for _ in range(max_iter):
        sample_indices = np.random.choice(num_points, 5, replace=False)
        sampled_points = points[sample_indices]

        if not check_pentagon(sampled_points):
            continue

        pentagon = sampled_points
        vertices = sampled_points[:10].reshape((5, 2))

        distances = np.array([point_to_pentagon_distance(point, vertices) for point in points])
        outliers = points[distances > threshold]
        inliers = points[distances <= threshold]

        if outliers is None:
            return pentagon, best_inliers, best_outliers

        if len(inliers) >= min_inliers:
            if len(inliers) > len(best_inliers):
                best_pentagon = pentagon
                best_inliers = inliers
                best_outliers = outliers

    return best_pentagon, best_inliers, best_outliers

def check_pentagon(vertices, angle_tolerance=3, side_tolerance=0.02):
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        angle_rad = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def distance(v1, v2):
        return np.linalg.norm(v1 - v2)

    angles = []
    for i in range(5):
        v1 = vertices[(i - 1) % 5] - vertices[i]
        v2 = vertices[(i + 1) % 5] - vertices[i]
        angles.append(angle_between_vectors(v1, v2))

    if not all(abs(angle - 108) <= angle_tolerance for angle in angles):
        return False

    side_lengths = [distance(vertices[i], vertices[(i + 1) % 5]) for i in range(5)]

    if not np.all(np.abs(side_lengths - side_lengths[0]) <= side_tolerance):
        return False

    return True

def point_to_pentagon_distance(point, vertices):
    edge_distances = []
    for i in range(5):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 5]
        edge_distances.append(point_to_line_distance(point, p1, p2))

    return np.min(edge_distances)

def point_to_line_distance(point, p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    num = np.abs(y_diff * point[0] - x_diff * point[1] + p2[0] * p1[1] - p2[1] * p1[0])
    denom = np.sqrt(y_diff ** 2 + x_diff ** 2)
    return num / denom
