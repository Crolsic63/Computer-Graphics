import numpy as np
from PIL import Image

def barr(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)) 
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)) 
    lambda2 = 1.0 - lambda0 - lambda1 
    return (lambda0, lambda1, lambda2)

def calculate_normal(v0, v1, v2):
    U = np.array([v1[i] - v0[i] for i in range(3)])
    V = np.array([v2[i] - v0[i] for i in range(3)])
    n = np.cross(U, V)
    return n / np.linalg.norm(n) if np.linalg.norm(n) != 0 else n

def draw_tr(image, z_buffer, v0, v1, v2, color):
    x0, y0, z0 = v0
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    
    xmin = int(min(x0, x1, x2))
    ymin = int(min(y0, y1, y2))
    xmax = int(max(x0, x1, x2))
    ymax = int(max(y0, y1, y2))

    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            lambdas = barr(x, y, x0, y0, x1, y1, x2, y2)
            lambda0, lambda1, lambda2 = lambdas

            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2 
                
                if z < z_buffer[x, y]: 
                    z_buffer[x, y] = z
                    image[x, y] = color

def read_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v'):
                parts = line.split()
                if len(parts) >= 4:  
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((9000 * x + 500, 9000 * y, 9000 * z))
    return vertices

def read_polygons(file_path):
    polygons = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('f'):
                parts = line.split()
                verts = [int(part.split('/')[0]) for part in parts[1:]]
                if len(verts) >= 3:
                    polygons.append(verts)
    return polygons

file_path = "C:\\Python_projects\\model_1.obj"
vertices = read_vertices(file_path)
polygons = read_polygons(file_path)
image_size = (1000, 1000, 3)
image = np.zeros(image_size, dtype=np.uint8)
z_buffer = np.zeros((1000, 1000))+np.inf  

light_direction = np.array([0, 0, 1])  

for polygon in polygons:
    for i in range(1, len(polygon) - 1):
        v0 = vertices[polygon[0] - 1]
        v1 = vertices[polygon[i] - 1]
        v2 = vertices[polygon[i + 1] - 1]

        normal = calculate_normal(v0, v1, v2)

        cos_angle = np.dot(normal, light_direction)

        if cos_angle < 0:  
            intensity = -255 * cos_angle  
            color = [intensity, intensity, intensity]  
            draw_tr(image, z_buffer, v0, v1, v2, color)

output_image = Image.fromarray(image, mode='RGB')
output_image = output_image.rotate(90, 1)
output_image.save('model_final.png')
output_image.show()