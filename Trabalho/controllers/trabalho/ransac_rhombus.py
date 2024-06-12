import numpy as np
import copy

def fit_rhombus_ransac(points, max_iter=400000, threshold=0.01, min_inliers=700):
    best_rhombus = None
    best_inliers = []
    best_outliers = copy.copy(points)
    num_points = len(points)

    for _ in range(max_iter):
        sample_indices = np.random.choice(num_points, 4, replace=False)
        sampled_points = points[sample_indices]

        if not check_rhombus(sampled_points):
            continue

        rhombus = sampled_points
        vertices = sampled_points[:8].reshape((4, 2))

        distances = np.array([point_to_rhombus_distance(point, vertices) for point in points])
        outliers = points[distances > threshold]
        inliers = points[distances <= threshold]

        if outliers is None:
            return rhombus, best_inliers, best_outliers

        if len(inliers) >= min_inliers:
            if len(inliers) > len(best_inliers):
                best_rhombus = rhombus
                best_inliers = inliers
                best_outliers = outliers

    return best_rhombus, best_inliers, best_outliers

def check_rhombus(vertices, side_tolerance=0.1, angle_tolerance=6, max_side_length=0.15):
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

    if not (abs(angles[0] - angles[2]) <= angle_tolerance and abs(angles[1] - angles[3]) <= angle_tolerance):
        return False

    side_lengths = np.array([
        distance(vertices[0], vertices[1]),
        distance(vertices[1], vertices[2]),
        distance(vertices[2], vertices[3]),
        distance(vertices[3], vertices[0])
    ])

    if not np.all(np.abs(side_lengths - side_lengths[0]) <= side_tolerance * side_lengths[0]) or not np.all(side_lengths <= max_side_length):
        return False

    return True

def point_to_rhombus_distance(point, vertices):
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
