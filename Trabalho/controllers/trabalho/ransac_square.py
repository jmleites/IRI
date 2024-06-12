import numpy as np
import copy

def fit_square_ransac(points, max_iter=1000000, threshold=0.01, min_inliers=500):
    best_square = None
    best_inliers = []
    best_outliers = copy.copy(points)
    num_points = len(points)

    for _ in range(max_iter):
        sample_indices = np.random.choice(num_points, 4, replace=False)
        sampled_points = points[sample_indices]

        if not check_square(sampled_points):
            continue

        square = sampled_points
        vertices = sampled_points[:8].reshape((4, 2))

        distances = np.array([point_to_square_distance(point, vertices) for point in points])
        outliers = points[distances > threshold]
        inliers = points[distances <= threshold]

        if outliers is None:
            return square, best_inliers, best_outliers

        if len(inliers) >= min_inliers:
            if len(inliers) > len(best_inliers):
                best_square = square
                best_inliers = inliers
                best_outliers = outliers

    return best_square, best_inliers, best_outliers

def check_square(vertices, side_tolerance=0.03, max_side_length=0.15):
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        angle_rad = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def distance(v1, v2):
        return np.linalg.norm(v1 - v2)

    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[1]
    v3 = vertices[3] - vertices[2]
    v4 = vertices[0] - vertices[3]

    angles = [
        angle_between_vectors(v1, -v4),
        angle_between_vectors(v2, -v1),
        angle_between_vectors(v3, -v2),
        angle_between_vectors(v4, -v3)
    ]

    if not are_points_ccw(vertices):
        return False

    for angle in angles:
        if not (86 <= angle <= 94):
            return False

    side_lengths = np.array([
        distance(vertices[0], vertices[1]),
        distance(vertices[1], vertices[2]),
        distance(vertices[2], vertices[3]),
        distance(vertices[3], vertices[0])
    ])
    if not np.all(np.abs(side_lengths - side_lengths[0]) <= side_tolerance) or not np.all(side_lengths <= max_side_length):
        return False

    return True

def are_points_ccw(vertices):
    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[1]
    v3 = vertices[3] - vertices[2]
    v4 = vertices[0] - vertices[3]

    cross_products = [
        cross_product(v1, -v4),
        cross_product(v2, -v1),
        cross_product(v3, -v2),
        cross_product(v4, -v3)
    ]

    return np.all(np.sign(cross_products) == np.sign(cross_products[0])) or np.all(cross_products == 0)

def point_to_square_distance(point, vertices):
    edge_distances = []
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 4]
        edge_distances.append(point_to_finite_line_distance(point, p1, p2))

    return np.min(edge_distances)


def point_to_finite_line_distance(point, p1, p2):
    if np.all(p1 == p2):
        return np.linalg.norm(point - p1)

    line_vec = p2 - p1
    pnt_vec = point - p1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    pnt_vec_scaled = pnt_vec / line_len
    t = np.dot(line_unitvec, pnt_vec_scaled)
    t = np.clip(t, 0, 1)
    nearest = line_vec * t
    dist = np.linalg.norm(nearest - pnt_vec)
    return dist

