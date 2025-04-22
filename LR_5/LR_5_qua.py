import math
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

def draw_tr(image, z_buffer, v0, v1, v2, I0, I1, I2, vt0, vt1, vt2, texture_img):
    x0, y0, z0 = v0
    x1, y1, z1 = v1
    x2, y2, z2 = v2

    xmin = max(0,int(min(x0,x1,x2)))
    ymin = max(0,int(min(y0,y1,y2)))
    xmax = min(1499, int(max(x0, x1, x2)))
    ymax = min(1499, int(max(y0, y1, y2)))

    W_T, H_T = texture_img.size

    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            lambdas = barr(x, y, x0, y0, x1, y1, x2, y2)
            lambda0, lambda1, lambda2 = lambdas

            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2

                if z < z_buffer[x, y]:
                    z_buffer[x, y] = z

                    u_t = lambda0 * vt0[0] + lambda1 * vt1[0] + lambda2 * vt2[0]
                    v_t = lambda0 * vt0[1] + lambda1 * vt1[1] + lambda2 * vt2[1]

                    tex_x = int(W_T * u_t)
                    tex_y = int(H_T * (1-v_t))
                    tex_color = texture_img.getpixel((tex_x, tex_y))

                    intensity = (lambda0 * I0 + lambda1 * I1 + lambda2 * I2)
                    if intensity <= 0:
                        image[x, y] = [int(c * (-intensity)) for c in tex_color]

def read_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
    return vertices

def read_texture_coords(file_path):
    texture_coords = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('vt '):
                parts = line.split()
                u, v = float(parts[1]), float(parts[2])
                texture_coords.append((u, v))
    return texture_coords

def read_polygons(file_path):
    polygons = []
    texture_indices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('f'):
                parts = line.split()
                verts = []
                tex_coords = []

                for part in parts[1:]:
                    elements = part.split('/')
                    verts.append(int(elements[0]))
                    if len(elements) > 1 and elements[1]:
                        tex_coords.append(int(elements[1]))
                    else:
                        tex_coords.append(None)

                # if len(verts) == 4:
                #     polygons.append([verts[0], verts[1], verts[2]])
                #     polygons.append([verts[0], verts[2], verts[3]])
                #
                #     texture_indices.append([tex_coords[0], tex_coords[1], tex_coords[2]])
                #     texture_indices.append([tex_coords[0], tex_coords[2], tex_coords[3]])
                # elif len(verts) >= 3:
                #     polygons.append(verts[:3])
                #     texture_indices.append(tex_coords[:3])

                if len(verts) >= 3:
                    for i in range(1, len(verts)-1):
                        polygons.append([verts[0],verts[i], verts[i+1]])
                        texture_indices.append([tex_coords[0], tex_coords[i], tex_coords[i+1]])

    return polygons, texture_indices

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

def quaternion_from_angle(axis, angle):
    angle_in_rad = np.radians(angle)
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half_angle = angle_in_rad / 2
    w = np.cos(half_angle)
    x, y, z = axis * np.sin(half_angle)
    return np.array([w, x, y, z])

def rotate_qua(vertices, axis, angle, tx, ty, tz):
    q = quaternion_from_angle(axis, angle)
    q = q / np.linalg.norm(q)
    q_w, q_x, q_y, q_z = q

    q_conj = np.array([q_w, -q_x, -q_y, -q_z])

    T = np.array([tx, ty, tz])
    transformed_vertices = []

    for v in vertices:
        v_x, v_y, v_z = v

        qv = np.array([0, v_x, v_y, v_z])

        temp_w = q_w * qv[0] - q_x * qv[1] - q_y * qv[2] - q_z * qv[3]
        temp_x = q_w * qv[1] + q_x * qv[0] + q_y * qv[3] - q_z * qv[2]
        temp_y = q_w * qv[2] - q_x * qv[3] + q_y * qv[0] + q_z * qv[1]
        temp_z = q_w * qv[3] + q_x * qv[2] - q_y * qv[1] + q_z * qv[0]

        rotated_w = temp_w * q_conj[0] - temp_x * q_conj[1] - temp_y * q_conj[2] - temp_z * q_conj[3]
        rotated_x = temp_w * q_conj[1] + temp_x * q_conj[0] + temp_y * q_conj[3] - temp_z * q_conj[2]
        rotated_y = temp_w * q_conj[2] - temp_x * q_conj[3] + temp_y * q_conj[0] + temp_z * q_conj[1]
        rotated_z = temp_w * q_conj[3] + temp_x * q_conj[2] - temp_y * q_conj[1] + temp_z * q_conj[0]

        v_rotated = np.array([rotated_x, rotated_y, rotated_z])

        v_transformed = v_rotated + T
        transformed_vertices.append(tuple(v_transformed))

    return transformed_vertices

def project(vertices, xScale, yScale, shiftX, shiftY):
    for i in range(0, len(vertices)):
        v = vertices[i]
        vertices[i] = (xScale * (v[0] / v[2]) + shiftX, yScale * (v[1] / v[2]) + shiftY, v[2])
    return vertices

def calculate_normals(polygons, vertices):
    vertex_normals = [np.zeros(3) for x in range(len(vertices))]

    for polygon in polygons:
        if (len(polygon) < 3):
            continue
        v0 = vertices[polygon[0] - 1]
        v1 = vertices[polygon[1] - 1]
        v2 = vertices[polygon[2] - 1]

        normal = calculate_normal(v0, v1, v2)

        for vertex in polygon:
            vertex_normals[vertex - 1] += normal

    for i in range(len(vertex_normals)):
        norm = np.linalg.norm(vertex_normals[i])
        if norm > 0:
            vertex_normals[i] /= norm
    return vertex_normals

def render_model(file_path, texture_path, image, z_buffer, alpha=0, betta=0, gamma=0, tx=0, ty=0, tz=1,xScale=9000, yScale=9000, shiftX=750, shiftY=750, use_quaternion=False):
    vertices = read_vertices(file_path)
    texture_coords = read_texture_coords(file_path)
    polygons, texture_indices = read_polygons(file_path)
    axis=[0, 1, 0]
    angle=60
    texture_img = Image.open(texture_path)

    if use_quaternion==True:
        vertices = rotate_qua(vertices, axis, angle, tx, ty, tz)
    else:
        vertices = rotate(vertices, alpha, betta, gamma, tx, ty, tz)

    vertex_normals = calculate_normals(polygons, vertices)
    vertices = project(vertices, xScale, yScale, shiftX, shiftY)

    light_direction = np.array([0, 0, 1])

    for poly_idx, polygon in enumerate(polygons):
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

            vt0 = texture_coords[texture_indices[poly_idx][0] - 1]
            vt1 = texture_coords[texture_indices[poly_idx][i] - 1]
            vt2 = texture_coords[texture_indices[poly_idx][i + 1] - 1]

            draw_tr(image, z_buffer, v0, v1, v2, I0, I1, I2, vt0, vt1, vt2, texture_img)

    return image, z_buffer

if __name__ == '__main__':
    image_size = (1500, 1500, 3)
    image = np.zeros(image_size, dtype=np.uint8)
    z_buffer = np.zeros((1500, 1500)) + np.inf

    image, z_buffer = render_model("C:\\Python_projects\\model_1.obj", "C:\\Python_projects\\tex.jpeg", image, z_buffer,alpha=pi/6, betta=pi, gamma=0, tx=0.12, ty=-0.04, tz=1,xScale=4000, yScale=4000, shiftX=750, shiftY=750, use_quaternion=True)
    # image, z_buffer = render_model("C:\\Python_projects\\model_1.obj","C:\\Python_projects\\tex.jpeg",image, z_buffer,alpha=0, betta=pi, gamma=0, tx=-0.07, ty=-0.04, tz=1, xScale=1000, yScale=1000, shiftX=750, shiftY=750)
    # image, z_buffer = render_model("C:\\Python_projects\\model_1.obj","C:\\Python_projects\\tex.jpeg",image, z_buffer,alpha=0, betta=pi/2, gamma=0, tx=-0.07, ty=-0.3, tz=1, xScale=2000, yScale=1000, shiftX=750, shiftY=750)
    # image, z_buffer = render_model("C:\\Python_projects\\model_1.obj","C:\\Python_projects\\tex.jpeg",image, z_buffer,alpha=0, betta=pi/3, gamma=0, tx=-0.07, ty=-0.4, tz=1, xScale=3000, yScale=1000, shiftX=750, shiftY=750)
    # image, z_buffer = render_model("C:\\Python_projects\\frog.obj","C:\\Python_projects\\12268_banjofrog_diffuse.jpg",image, z_buffer,alpha=0, betta=pi, gamma=0,tx=-0.07, ty=-0.04, tz=5,xScale=350, yScale=350,shiftX=750, shiftY=750,use_quaternion=True)
    # image, z_buffer = render_model("C:\\Python_projects\\cat.obj", "C:\\Python_projects\\cat.jpg",image, z_buffer, alpha = pi / 3, betta=0, gamma=0, tx=0.07, ty=-2, tz=10, xScale=300,yScale=300, shiftX=750, shiftY=750,use_quaternion=True)
    image, z_buffer = render_model("C:\\Python_projects\\frog.obj", "C:\\Python_projects\\12268_banjofrog_diffuse.jpg",image, z_buffer, alpha=pi / 2, betta=0, gamma=pi, tx=0.07, ty=-2, tz=5, xScale=500,yScale=500, shiftX=750, shiftY=750)
    # image, z_buffer = render_model("C:\\Python_projects\\model.obj","C:\\Python_projects\\model.bmp",image, z_buffer,alpha=0, betta=pi, gamma=0, tx=1, ty=0.05, tz=2, xScale=700, yScale=700, shiftX=750, shiftY=750)
    # image, z_buffer = render_model("C:\\Python_projects\\frog.obj","C:\\Python_projects\\12268_banjofrog_diffuse.jpg",image, z_buffer,alpha=0, betta=pi, gamma=0,tx=-0.07, ty=-0.04, tz=5,xScale=150, yScale=150,shiftX=750, shiftY=750,use_quaternion=True)
    # image, z_buffer = render_model("C:\\Python_projects\\frog.obj","C:\\Python_projects\\12268_banjofrog_diffuse.jpg",image, z_buffer,alpha=0, betta=pi, gamma=0,tx=-0.07, ty=-0.05, tz=5,xScale=150, yScale=150,shiftX=750, shiftY=750,use_quaternion=True)

    output_image = Image.fromarray(image, mode='RGB')
    output_image = output_image.rotate(90)
    output_image.save('model_34.png')
    output_image.show()