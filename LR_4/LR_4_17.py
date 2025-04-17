import numpy as np
from math import pi
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

def draw_tr(image, z_buffer, v0, v1, v2, I0, I1, I2):
    x0, y0, z0 = v0
    x1, y1, z1 = v1
    x2, y2, z2 = v2

    xmin = int(min(x0, x1, x2))
    ymin = int(min(y0, y1, y2))
    xmax = int(max(x0, x1, x2))
    ymax = int(max(y0, y1, y2))

    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            lambdas = barr(x, y, x0, y0, x1, y1, x2, y2)
            lambda0, lambda1, lambda2 = lambdas

            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2

                if z < z_buffer[x, y]:
                    z_buffer[x, y] = z
                    intensity = -255 * (lambda0*I0+lambda1*I1+lambda2*I2)
                    if ((lambda0*I0+lambda1*I1+lambda2*I2)<=0):
                        image[x, y] = [intensity, intensity, intensity]

def read_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
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

def rotate(vertices, alpha, betta, gamma, tx, ty, tz):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), np.sin(alpha)],
        [0, -np.sin(alpha), np.cos(alpha)]
    ])
    R_y = np.array([
        [np.cos(betta), 0, np.sin(betta)],
        [0, 1, 0],
        [-np.sin(betta), 0, np.cos(betta)]
    ])
    R_z = np.array([
        [np.cos(gamma), np.sin(gamma), 0],
        [-np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    R = np.dot(np.dot(R_x, R_y), R_z)
    T = np.array([tx, ty, tz])
    trans_vertices = [np.dot(R, vertex[:3]) + T for vertex in vertices]
    return trans_vertices

def project(vertices, xScale, yScale, shiftX, shiftY):
    for i in range(0, len(vertices)):
        v = vertices[i]
        vertices[i] = (xScale * (v[0] / v[2]) + shiftX, yScale * (v[1] / v[2]) + shiftY, v[2])
    return vertices

def calculate_normals(polygons,vertices):
    vertex_normals = [np.zeros(3) for x in range(len(vertices))]

    for polygon in polygons:
        if (len(polygon) < 3):
            continue
        v0=vertices[polygon[0]-1]
        v1=vertices[polygon[1]-1]
        v2=vertices[polygon[2]-1]

        normal=calculate_normal(v0, v1, v2)

        for vertex in polygon:
            vertex_normals[vertex - 1] += normal

    for i in range(len(vertex_normals)):
        norm=np.linalg.norm(vertex_normals[i])
        if norm > 0:
            vertex_normals[i] /= norm
    return vertex_normals

file_path = "C:\\Python_projects\\model_1.obj"
vertices = read_vertices(file_path)
polygons = read_polygons(file_path)

alpha = 0
betta = pi
gamma = 0
tx = 0
ty = -0.04
tz = 1
vertices = rotate(vertices, alpha, betta, gamma, tx, ty, tz)
oldVertices = vertices.copy()
vertex_normals = calculate_normals(polygons, vertices)
vertices = project(vertices, 9000, 9000, 1500 / 2, 1500 / 2)
image_size = (1500, 1500, 3)
image = np.zeros(image_size, dtype=np.uint8)
z_buffer = np.zeros((1500, 1500)) + np.inf

light_direction = np.array([0, 0, 1])

for polygon in polygons:
    for i in range(1, len(polygon) - 1):
        v0 = vertices[polygon[0] - 1]
        v1 = vertices[polygon[i] - 1]
        v2 = vertices[polygon[i + 1] - 1]

        n0 = vertex_normals[polygon[0] - 1]
        n1 = vertex_normals[polygon[i] - 1]
        n2 = vertex_normals[polygon[i + 1] - 1]

        I0 = np.dot(n0, light_direction)
        I1 = np.dot(n1, light_direction)
        I2 = np.dot(n2, light_direction)

        draw_tr(image, z_buffer, v0, v1, v2, I0, I1, I2)

output_image = Image.fromarray(image, mode='RGB')
output_image = output_image.rotate(90)
output_image.save('model_1.png')
output_image.show()