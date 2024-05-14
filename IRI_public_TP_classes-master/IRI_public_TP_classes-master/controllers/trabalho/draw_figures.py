import numpy as np
from matplotlib import pyplot as plt

def draw_circle(circle_center: tuple, circle_radius: float) -> None:
    angles = np.linspace(0, 2 * np.pi, 100)

    circle_x = circle_center[0] + circle_radius * np.cos(angles)
    circle_y = circle_center[1] + circle_radius * np.sin(angles)

    plt.plot(circle_x, circle_y, color='blue', label='Circle', linewidth=3)
    plt.scatter(circle_center[0], circle_center[1], color='blue', label='Center', marker='o')

def draw_triangle(vertices: np.ndarray) -> None:
    vertices = np.vstack([vertices, vertices[0]])

    plt.plot(vertices[:, 0], vertices[:, 1], color='blue', label='Triangle', linewidth=3)
    plt.scatter(vertices[:, 0], vertices[:, 1], color='blue', label='Vertices', marker='o')
