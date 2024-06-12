import os
import math
from PIL import Image, ImageDraw
import numpy as np
import random
import csv
from math import cos, sin, radians

def rotate_point(x, y, cx, cy, angle):
    rad = radians(angle)
    cos_a = cos(rad)
    sin_a = sin(rad)
    dx = x - cx
    dy = y - cy
    new_x = cx + dx * cos_a - dy * sin_a
    new_y = cy + dx * sin_a + dy * cos_a
    return new_x, new_y

def create_random_map(image_path):
    width = 700
    height = 700
    image = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(image)
    shapes = []
    centroids_and_orientations = []
    resolution = 0.001

    def does_overlap(new_shape):
        for shape in shapes:
            if not (
                    new_shape[0] >= shape[2] or new_shape[2] <= shape[0] or
                    new_shape[1] >= shape[3] or new_shape[3] <= shape[1]
            ):
                return True
        return False

    def is_within_bounds(corners, width, height):
        for x, y in corners:
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        return True

    def get_pentagon_corners(cx, cy, radius):
        corners = []
        for i in range(5):
            theta = math.radians(i * 72)
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            corners.append((x, y))
        return corners

    shape_types = ['square', 'circle', 'triangle', 'diamond', 'pentagon']
    for shape_type in shape_types:
        tries = 0
        max_tries = 50
        while tries < max_tries:
            tries += 1
            if shape_type == 'square':
                rect_width = random.randint(120, 130)
                rect_height = rect_width
                x1 = random.randint(0, width - rect_width)
                y1 = random.randint(0, height - rect_height)
                angle = random.randint(0, 360)
                cx = x1 + rect_width / 2
                cy = y1 + rect_height / 2

                corners = [
                    rotate_point(x1, y1, cx, cy, angle),
                    rotate_point(x1 + rect_width, y1, cx, cy, angle),
                    rotate_point(x1 + rect_width, y1 + rect_height, cx, cy, angle),
                    rotate_point(x1, y1 + rect_height, cx, cy, angle)]

                new_shape = [min(c[0] for c in corners), min(c[1] for c in corners), max(c[0] for c in corners),
                             max(c[1] for c in corners)]
                if not does_overlap(new_shape) and is_within_bounds(corners, width, height):
                    shapes.append(new_shape)
                    draw.polygon(corners, outline=0, width=3)
                    vertices = []
                    for vx, vy in corners:
                        real_vx = vx * resolution
                        real_vy = height * resolution - (vy * resolution)
                        vertices.append((real_vx, real_vy))
                    centroids_and_orientations.append((shape_type, cx, cy, vertices, angle))
                    break

            elif shape_type == 'circle':
                radius = random.randint(70, 80)
                x1 = random.randint(radius, width - radius)
                y1 = random.randint(radius, height - radius)
                new_radius = radius * resolution
                new_shape = [x1 - radius, y1 - radius, x1 + radius, y1 + radius]
                if not does_overlap(new_shape):
                    shapes.append(new_shape)
                    draw.ellipse(new_shape, outline=0, width=3)
                    centroids_and_orientations.append((shape_type, x1, y1, new_radius, 0))
                    break

            elif shape_type == 'triangle':
                base = random.randint(100, 120)
                height_t = base * (3 ** 0.5) / 2
                x1 = random.randint(0, width - base)
                y1 = random.randint(0, height - int(height_t))
                x2 = x1 + base
                y2 = y1
                x3 = x1 + base // 2
                y3 = y1 + int(height_t)
                angle = random.randint(0, 360)
                cx = (x1 + x2 + x3) / 3
                cy = (y1 + y2 + y3) / 3

                corners = [
                    rotate_point(x1, y1, cx, cy, angle),
                    rotate_point(x2, y2, cx, cy, angle),
                    rotate_point(x3, y3, cx, cy, angle)
                ]

                new_shape = [min(c[0] for c in corners), min(c[1] for c in corners), max(c[0] for c in corners),
                             max(c[1] for c in corners)]
                if not does_overlap(new_shape) and is_within_bounds(corners, width, height):
                    shapes.append(new_shape)
                    draw.polygon(corners, outline=0, width=3)
                    vertices = []
                    for vx, vy in corners:
                        real_vx = vx * resolution
                        real_vy = height * resolution - (vy * resolution)
                        vertices.append((real_vx, real_vy))
                    centroids_and_orientations.append((shape_type, cx, cy, vertices, angle))
                    break


            elif shape_type == 'diamond':
                diamond_width = random.randint(100, 120)
                diamond_height = random.randint(150, 200)
                x1 = random.randint(0, width - diamond_width)
                y1 = random.randint(0, height - diamond_height)
                angle = random.randint(0, 360)
                cx = x1 + diamond_width / 2
                cy = y1 + diamond_height / 2

                corners = [
                    rotate_point(cx, y1, cx, cy, angle),
                    rotate_point(x1 + diamond_width, cy, cx, cy, angle),
                    rotate_point(cx, y1 + diamond_height, cx, cy, angle),
                    rotate_point(x1, cy, cx, cy, angle)
                ]
                new_shape = [min(c[0] for c in corners), min(c[1] for c in corners),
                             max(c[0] for c in corners), max(c[1] for c in corners)]
                if not does_overlap(new_shape) and is_within_bounds(corners, width, height):
                    shapes.append(new_shape)
                    draw.polygon(corners, outline=0, width=3)
                    vertices = []
                    for vx, vy in corners:
                        real_vx = vx * resolution
                        real_vy = height * resolution - (vy * resolution)
                        vertices.append((real_vx, real_vy))
                    centroids_and_orientations.append((shape_type, cx, cy, vertices, angle))
                    break

            elif shape_type == 'pentagon':
                radius = random.randint(70, 80)
                cx = random.randint(radius, width - radius)
                cy = random.randint(radius, height - radius)
                angle = random.randint(0, 360)
                corners = get_pentagon_corners(cx, cy, radius)
                rotated_corners = [rotate_point(x, y, cx, cy, angle) for x, y in corners]
                new_shape = [min(c[0] for c in rotated_corners), min(c[1] for c in rotated_corners),
                             max(c[0] for c in rotated_corners), max(c[1] for c in rotated_corners)]

                if not does_overlap(new_shape) and is_within_bounds(rotated_corners, width, height):
                    shapes.append(new_shape)
                    draw.polygon(rotated_corners, outline=0, width=3)
                    vertices = []
                    for vx, vy in corners:
                        real_vx = vx * resolution
                        real_vy = height * resolution - (vy * resolution)
                        vertices.append((real_vx, real_vy))
                    centroids_and_orientations.append((shape_type, cx, cy, vertices, angle))
                    break

            tries += 1

    image.save(image_path)
    print(f"Random map image saved to {image_path}")

    centroids_file_path = os.path.splitext(image_path)[0] + "_centroids.txt"
    resolution = 0.001
    with open(centroids_file_path, 'w') as f:
        for shape_type, cx, cy, vertice, angle in centroids_and_orientations:
            real_x = cx * resolution
            real_y = height * resolution - (cy * resolution)
            centroid = [real_x, real_y]
            f.write(f'{shape_type} = {centroid}, {vertice} ,{angle}\n')
    print(f"Centroids and orientations saved to {centroids_file_path}")


def create_yaml_data(image_name, resolution=0.001, origin=[0.0, 0.0, 0.0], occupied_thresh=0.65):
    yaml_data = {
        'image': image_name,
        'resolution': resolution,
        'origin': origin,
        'occupied_thresh': occupied_thresh
    }
    return yaml_data


def process_image_and_generate_map(custom_maps_filepath, yaml_data):
    image_filename = yaml_data['image']
    resolution = yaml_data['resolution']
    origin = yaml_data['origin']
    occupied_thresh = yaml_data['occupied_thresh']
    max_pixel_value_for_wall = int(255 * occupied_thresh)
    wall_pixels_coords = []

    img = Image.open(os.path.join(custom_maps_filepath, image_filename)).convert('L')
    np_img = np.array(img)
    height, width = np_img.shape

    for row in range(height):
        for col in range(width):
            if np_img[row][col] <= max_pixel_value_for_wall:
                wall_pixels_coords.append((origin[0] + resolution * col, origin[1] + resolution * (height - row)))

    print('Number of walls:', len(wall_pixels_coords))

    map_name = os.path.splitext(image_filename)[0]
    points_file_path = os.path.join(custom_maps_filepath, f"{map_name}_points.csv")
    with open(points_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for x in range(width):
            writer.writerow((origin[0] + resolution * x, origin[1]))
            writer.writerow((origin[0] + resolution * x, origin[1] + resolution * (height - 1)))
        for y in range(1, height - 1):
            writer.writerow((origin[0], origin[1] + resolution * y))
            writer.writerow((origin[0] + resolution * (width - 1), origin[1] + resolution * y))
        for coord in wall_pixels_coords:
            writer.writerow(coord)

    base_map_webots_filepath = os.path.join(custom_maps_filepath, 'base_map.wbt')
    with open(base_map_webots_filepath, 'r') as f:
        webots_str = f.read()

    map_webots_filepath = os.path.join(custom_maps_filepath, f"{map_name}.wbt")
    with open(map_webots_filepath, 'w') as f:
        f.write(webots_str)

        f.write('RectangleArena {\n')
        f.write(f'  translation {origin[0] + resolution * width / 2} {origin[1] + resolution * height / 2} 0.0\n')
        f.write(f'  floorSize {resolution * width} {resolution * height}\n')
        f.write('  floorTileSize 0.25 0.25\n')
        f.write('  floorAppearance Parquetry {\n')
        f.write('    type "light strip"\n')
        f.write('  }\n')
        f.write('  wallHeight 0.05\n')
        f.write('}\n')

        for index, coord in enumerate(wall_pixels_coords):
            f.write('Solid {\n')
            f.write(f'    translation {coord[0]} {coord[1]} 0.025\n')
            f.write('    children [\n')
            f.write('        Shape {\n')
            f.write('            geometry Box {\n')
            f.write(f'                size {resolution} {resolution} 0.05\n')
            f.write('            }\n')
            f.write('        }\n')
            f.write('    ]\n')
            f.write(f'    name "solid{index}"\n')
            f.write('}\n')


if __name__ == '__main__':
    custom_maps_filepath = os.getcwd()
    image_name = 'random_map.png'

    create_random_map(os.path.join(custom_maps_filepath, image_name))

    yaml_data = create_yaml_data(image_name)

    process_image_and_generate_map(custom_maps_filepath, yaml_data)
