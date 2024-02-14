# computes which vertices of a mesh are "valid"
# we don't want any valleys that are "untouchable" by a human finger

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import math
import numpy as np

# FINGER_CIRCUMFERENCE = 75 # mm
# FINGER_RADIUS = FINGER_CIRCUMFERENCE / (2 * math.pi)
# FINGER_RADIUS = 7.5 #mm
FINGER_RADIUS = 10 #mm

class VertexLite:
    def __init__(self, index, co, select=False):
        self.index = index
        self.co = co
        self.select = select

class InvalidVertex:
    def __init__(self, index, angle_from_center):
        self.index = index # index of the vertex in the vert array
        self.angle_from_center = angle_from_center # angle the vertex made with the original

# 2.5d cylinder intersection going UP from a particular center point
def intersects_cylinder(cyl_bottom_center: Vector, cyl_radius: float, point: Vector) -> bool:
    # point_radius = (cyl_bottom_center.xy - point.xy).magnitude
    xdiff = cyl_bottom_center.x - point.x 
    ydiff = cyl_bottom_center.y - point.y 
    point_radius_sq = xdiff * xdiff + ydiff * ydiff
    return point_radius_sq < (cyl_radius * cyl_radius) and point.z > cyl_bottom_center.z

# find x-y spacing of a grid DEM (in local space), as well as grid width and height
def find_spacing(vertices):
    x_spacing = abs(vertices[1].co.x - vertices[0].co.x)

    width = 1
    y_coord = vertices[0].co.y
    y_coord_last = vertices[width].co.y
    while abs(y_coord - y_coord_last) < 0.001:
        y_coord_last = vertices[width].co.y
        width += 1

    print(y_coord, y_coord_last)
    y_spacing = abs(vertices[width].co.y - vertices[0].co.y)
    return (width - 1, len(vertices) // width, x_spacing, y_spacing)

def get_vert_neighbors(vert: bmesh.types.BMVert, exclued_indices: set) -> list:
    ret = []
    for e in vert.link_edges:
        v_neighbor = e.other_vert(vert)
        if v_neighbor.index not in exclued_indices:
            ret.append(v_neighbor)
    return ret

def map_range(value, low1, high1, low2, high2):
    '''
    remap a value from range (low1, high1) => (low2, high2)
    '''
    return low2 + (high2 - low2) * (value - low1) / (high1 - low1)

def main():
    heightmap = bpy.context.active_object
    # vertices = list(heightmap.data.vertices)

    HEIGHTMAP_BOUNDS = [Vector(heightmap.bound_box[i]) for i in range(8)]

    # assumes that heightmap is nicely (row-major) gridded, top-left is row 0, col 0
    # also assumes there aren't any vertices NOT on the grid
    # make them so
    # warning: vertices[] has a different index structure than heightmap.data.vertices
    vertices = [VertexLite(v.index, v.co, v.select) for v in heightmap.data.vertices]
    vertices.sort(key=lambda v: v.co.x)
    vertices.sort(key=lambda v: v.co.y, reverse=True)


    grid_width, grid_height, x_spacing, y_spacing = find_spacing(vertices)

    heightmap_bmesh = bmesh.new()
    heightmap_bmesh.from_mesh(heightmap.data)
    heightmap_bvh = BVHTree.FromBMesh(heightmap_bmesh)

    print('grid width', 'grid height', 'x spacing (m)', 'y spacing (m)')
    print(grid_width, grid_height, x_spacing, y_spacing)
    print('Checking vertices for touchability')
    x_spacing_world = x_spacing * heightmap.scale.x
    y_spacing_world = y_spacing * heightmap.scale.y

    # selected = next(filter(lambda v: v.select, vertices))
    # selected = None
    # selected_i = 0
    # for i in range(len(vertices)):
    #     if vertices[i].select:
    #         selected = vertices[i]
    #         selected_i = i
    # print('selected', selected.index, selected_i)

    debug = []
    all_invalid = set()
    # for v in [vertices[27264]]: # in steep valley
    # for v in [vertices[26723]]: # in wide valley
    # for v in [vertices[30399]]: # on a valid one-sided slope
    # for v in vertices:
    min_radius = 1 #mm
    rings = 3
    distance_per_sample = 1 # mm
    # for i, v in enumerate([selected]):
    for i, v in enumerate(vertices):
        v_world = heightmap.matrix_world @ v.co

        ring_validity = []
        for ring in range(1, rings + 1):
            ring_radius = map_range(ring, 0, rings, min_radius, FINGER_RADIUS)

            # define a paraboloid to approximate a finger.
            # should be at FINGER RADIUS by the time it gets to the outer ring
            paraboloid_z = v_world.z + (1.0 / FINGER_RADIUS) * ring_radius * ring_radius

            # iterate around a circle and find "runs" where the sampled height
            # is BELOW the finger (not intersecting the paraboloid)
            sample_valid = []
            circumference = 2 * math.pi * ring_radius
            samples = circumference / distance_per_sample
            t = 0
            while t < math.pi * 2:
                # construct circle
                vx = math.cos(t) * ring_radius
                vy = math.sin(t) * ring_radius
                cxy = Vector((vx, vy, 0))
                cp = cxy + v_world
                cp_dem = heightmap.matrix_world.inverted() @ cp

                # sample /raycast to the surface to get z / vertical component
                loc, _normal, _index, _dist = heightmap_bvh.ray_cast(cp_dem + Vector((0, 0, HEIGHTMAP_BOUNDS[6].z)), Vector((0, 0, -1)))
                if loc is not None:
                    loc_world = heightmap.matrix_world @ loc
                    cp.z = loc_world.z
                    sample_valid.append((cp.z < paraboloid_z, cp, paraboloid_z))
                else:
                    # count samples past the edge of the map as "valid"
                    sample_valid.append((True, cp, paraboloid_z))
                # DEBUG
                # bpy.ops.mesh.primitive_uv_sphere_add(radius=0.20, enter_editmode=False, align='WORLD', location=cp, scale=(1, 1, 1))
                # sph = bpy.context.active_object
                # sph.name = 'cp'
                # debug.append(sph)
                # bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, enter_editmode=False, align='WORLD', location=(cp.x, cp.y, paraboloid_z), scale=(1, 1, 1))
                # sph = bpy.context.active_object
                # sph.name = 'cp'
                # debug.append(sph)
                t += (math.pi * 2) / samples
            ring_validity.append(sample_valid)

        # check for "runs" of connected, valid samples
        run_lengths = []
        for ri, ring in enumerate(ring_validity):
            previous_point = None
            for valid, point, par_z in ring:
                if previous_point is None:
                    run_lengths.append(0)

                # instead, using arc length
                if valid and previous_point is not None:
                    segment_length = (point - previous_point).magnitude
                    run_lengths[-1] += segment_length

                if valid:
                    previous_point = point
                else:
                    previous_point = None

                # could use chords...
                # if valid and run_start_pt is None:
                #     run_start_pt = point
                # if not valid and run_start_pt is not None:
                #     run_dist_chord = (point - run_start_pt).magnitude
                #     run_lengths.append(run_dist_chord)
                #     run_start_pt = None
                # DEBUG
                # r = 0.2 if valid else 0.05
                # bpy.ops.mesh.primitive_uv_sphere_add(radius=r, enter_editmode=False, align='WORLD', location=point, scale=(1, 1, 1))
                # sph = bpy.context.active_object
                # sph.name = 'cp'
                # debug.append(sph)
        run_lengths = [r for r in run_lengths if r > 0]
        if len(run_lengths) > 0:
            avg_run_length = sum(run_lengths) / len(run_lengths)
            if avg_run_length < FINGER_RADIUS * 2:
                all_invalid.add(i)

        perc_int = 100 * v.index / len(vertices)
        if abs(perc_int - int(perc_int)) < 0.0001:
            print('Vertex {} / {} ({:.0f}%)'.format(v.index, len(vertices), perc_int))

    for i in all_invalid:
        vertices[i].select = True

    print('There were {} / {} invalid vertices ({:.1%})'.format(len(all_invalid), grid_height * grid_width, len(all_invalid) / (grid_width * grid_height)))

    # with bpy.context.temp_override(active_object=debug[0], selected_editable_objects=debug):
    #     bpy.ops.object.join()

    heightmap_bmesh.free()
    for i, v in enumerate(vertices):
        heightmap.data.vertices[i].select = v.select


if __name__ == '__main__':
    # exit(main())
    main()