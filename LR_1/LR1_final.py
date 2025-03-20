import numpy as np
from PIL import Image, ImageOps

def drawline(image, x0, y0, x1, y1, color):
    xchange = False
    if(abs(x0-x1)<abs(y0-y1)):
        x0,y0=y0,x0
        x1,y1=y1,x1
        xchange=True
    
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if (derror > (x1 - x0)):
            derror -=  2*(x1 - x0)
            y += y_update

def draw_model(polygons, vertices, image_size):
    image = np.full(image_size, 255, dtype=np.uint8)

    for polygon in polygons:
        for i in range(len(polygon)):
            v1 = vertices[polygon[i] - 1]
            v2 = vertices[polygon[(i + 1 ) % len(polygon)] - 1]
            x1, y1 = map(int, np.round(v1))
            x2, y2 = map(int, np.round(v2))
            drawline(image, x1, y1, x2, y2, 0)

    return image            

def read_vertices(file_path):
    vertices = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v'):
                parts = line.split()
                x, y = float(parts[1]), float(parts[2])
                vertices.append((9000*x+500, 9000*y))
    
    return vertices

def read_polygons(file_path):
    polygons = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('f'):
                parts = line.split()
                vertices = [int(part.split('/')[0]) for part in parts[1:]]
                polygons.append(vertices)
    return polygons

file_path = "C:\Python_projects\LR_1\model_1.obj"

vertices = read_vertices(file_path)
polygons = read_polygons(file_path)

image_size = (1000, 1000)
image = draw_model(polygons, vertices, image_size)

image = ImageOps.flip(Image.fromarray(image, mode='L'))
image.save('model.png')
image.show()