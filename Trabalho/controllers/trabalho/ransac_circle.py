import numpy as np
import copy

def fit_circle_ransac(points, max_iter=100000, threshold=0.01, min_inliers=1100, max_radius = 0.1):
    best_circle = None
    best_inliers = []
    best_outliers = copy.copy(points)
    num_points = len(points)

    for _ in range(max_iter):
        sample_indices = np.random.choice(num_points, 3, replace=False)
        sampled_points = points[sample_indices]

        circle = fit_circle(sampled_points)
        if circle is not None:
            center = circle[:2]
            radius = circle[2]

            distances = np.sqrt((points[:, 0] - center[0]) ** 2 + (points[:, 1] - center[1]) ** 2)
            outliers = points[np.abs(distances - radius) > threshold]
            inliers = points[np.abs(distances - radius) <= threshold]

            if outliers is None:
                return circle, best_inliers, best_outliers

            if len(inliers) > min_inliers and radius < max_radius:
                if len(inliers) > len(best_inliers):
                    best_circle = circle
                    best_inliers = inliers
                    best_outliers = outliers

    return best_circle, best_inliers, best_outliers

def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack((2 * x, 2 * y, np.ones(len(points))))
    b = x ** 2 + y ** 2
    center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    radius = np.sqrt(center[0] ** 2 + center[1] ** 2 + center[2])
    radius_squared = center[0] ** 2 + center[1] ** 2 + center[2]

    if center[0] - np.sqrt(radius_squared) < 0 or center[0] + np.sqrt(radius_squared) < 0 \
        or center[1] - np.sqrt(radius_squared) < 0 or center[1] + np.sqrt(radius_squared) < 0:
        return None

    return np.append(center[:2], radius)

