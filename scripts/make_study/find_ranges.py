import bpy
from mathutils import Vector
import json
import importlib
from pathlib import Path

class DataRange:
    def __init__(self, row: int, col: int, center_dem_x: float, center_dem_y: float, radius: float, min: float, max: float, mean: float) -> None:
        self.row = row
        self.col = col
        self.center_dem_x = center_dem_x
        self.center_dem_y = center_dem_y
        self.radius = radius
        self.min = min
        self.max = max
        self.mean = mean

    def __str__(self):
        return 'DataRange({}, {}, {:.1f}, {:.1f}, {:.1f}, {:.1f})'.format(self.row, self.col, self.max - self.min, self.min, self.max, self.mean)

    def to_json(self):
        return self.__dict__


def main():
    data = bpy.context.active_object
    data_bounds = [data.matrix_world @ Vector(b) for b in data.bound_box]

    selector_size = 150 # mm

    # geo coordinates of origin for geojson
    bgis = importlib.import_module("BlenderGIS-228")
    geo_scene = bgis.geoscene.GeoScene(bpy.context.scene)

    # selector = bpy.context.active_object

    # for v in verts_inside:
    #     data.data.vertices[v.index].select = True

    # data bounds/size in world coordinates
    data_upperleft = Vector(data_bounds[0])
    data_size = Vector(data_bounds[6]) - Vector(data_bounds[0])
    data_width = data_size.x
    data_height = data_size.y

    data_ranges = []
    num_rows = int(data_height / selector_size)
    num_cols = int(data_width / selector_size)
    for r in range(num_rows):
        y = data_upperleft.y + (data_height / num_rows) * r + selector_size / 2
        for c in range(num_cols):
            x = data_upperleft.x + (data_width / num_cols) * c + selector_size / 2
            p = Vector((x, y, 0))
            p_scaled = data.matrix_world.inverted() @ p
            x_dem, y_dem = geo_scene.view3dToProj(p_scaled.x, p_scaled.y)
            p_dem = Vector((x_dem, y_dem, 0))
            # selector_bounds = [selector.matrix_world @ Vector(b) + p for b in selector.bound_box]
            selector_bounds = [
                p - Vector((selector_size, selector_size, 0)) / 2.0,
                p, p, p, p, p,
                p + Vector((selector_size, selector_size, 0)) / 2.0,
                p
            ]
            verts_inside = [v for v in data.data.vertices if within_bounds(data.matrix_world @ v.co, selector_bounds, check_axes=(True, True, False))]
            # verts_inside = [v for v in data.data.vertices if within_radius(data.matrix_world @ v.co, p, selector_size / 2, check_axes=(True, True, False))]
            bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=p, scale=selector_size * Vector((1, 1, 2)))
            # bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1, enter_editmode=False, align='WORLD', location=p, scale=selector_size * Vector((1, 1, 1)))
            obj = bpy.context.active_object

            zs = [v.co.z for v in verts_inside]
            v_sum = sum(zs)
            v_len = len(zs)
            v_avg = v_sum / v_len
            v_min = min(zs)
            v_max = max(zs)

            # print('{}, {}:'.format(r, c))
            # print('- Min: ', v_min)
            # print('- Max: ', v_max)
            # print('- Range: ', v_max - v_min)
            print('progress: {} / {}'.format(1 + r * num_cols + c, num_rows * num_cols))
            obj.name = '{4:.0f}m ({5:.2f}) {0}, {1}: {2:.0f}m-{3:.0f}m'.format(r, c, v_min, v_max, v_max - v_min, (v_max - v_min) * data.scale.z)
            data_ranges.append(DataRange(r, c, p_dem.x, p_dem.y, (selector_size / 2) / data.scale.x, v_min, v_max, v_avg))
            
    data_ranges.sort(key=lambda dr: dr.max - dr.min)
    for dr in data_ranges:
        print(dr)
    file_name = data.name + '_ranges.json'
    save_path = Path('~/Documents/research/proposal/LineStudySources/{}/inter/{}'.format(data.name, file_name)).expanduser()
    if not save_path.parent.exists():
        save_path.parent.mkdir()

    with open(save_path, 'w') as fout:
        data_ranges_json = [dr.to_json() for dr in data_ranges]
        json.dump(data_ranges_json, fout, indent=4)

def within_radius(target, center, radius, check_axes=(True, True, True)):
    check_x, check_y, check_z = check_axes
    dx = (target.x - center.x)
    dy = (target.y - center.y)
    dz = (target.z - center.z)
    suma = 0
    if check_x:
        suma += dx * dx
    if check_y:
        suma += dy * dy
    if check_z:
        suma += dz * dz
    return suma < radius * radius

def within_bounds(point, bound_box, check_axes=(True, True, True)):
    '''
    Determine if point inside bounding box. Assumes blender-formatted bounding box like
    world bounds on a 2x2x2 cube (for example):
    (-1.0, -1.0, -1.0)
    (-1.0, -1.0, 1.0)
    (-1.0, 1.0, 1.0)
    (-1.0, 1.0, -1.0)
    (1.0, -1.0, -1.0)
    (1.0, -1.0, 1.0)
    (1.0, 1.0, 1.0)
    (1.0, 1.0, -1.0)

    Optionally only check for certain axes.
    '''
    check_x, check_y, check_z = check_axes
    x = point.x > bound_box[0].x and point.x < bound_box[6].x
    y = point.y > bound_box[0].y and point.y < bound_box[6].y
    z = point.z > bound_box[0].z and point.z < bound_box[6].z
    result = True
    if check_x:
        result = result and x
    if check_y:
        result = result and y
    if check_z:
        result = result and z
    return result

if __name__ == '__main__':
    main()