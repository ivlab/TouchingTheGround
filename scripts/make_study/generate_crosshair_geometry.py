import sys
import math
import json
import os
import bpy
from mathutils import Vector, Euler, Matrix
from mathutils.bvhtree import BVHTree
from pathlib import Path
import random
import importlib
import bmesh

index_to_letter = lambda i: chr(65 + i)

PATH = Path('~/Documents/research/proposal')

MAKE_GEOMETRY = True
MAKE_LABELS = True
MAKE_LEGEND = True
MAKE_FILE_OUTPUT = True

class LegendEntry:
    def __init__(self, value, scale, position):
        self.value = value
        self.scale = d_vec3(scale)
        self.position = d_vec3(position)

class VertexLite:
    def __init__(self, index,   co, select=False):
        self.index = index
        self.co = co
        self.select = select

class LinePoint:
    def __init__(self, coord: Vector, world_coord: Vector, index: int) -> None:
        self.coord = coord
        self.world_coord = world_coord
        self.index = index

    def to_json(self):
        return {
            'coord': d_vec3(self.coord),
            'world_coord': d_vec3(self.world_coord),
            'index': self.index
        }

    def __str__(self):
        return str(self.coord)

class AdvectAnswer:
    def __init__(self, line_index: int, point_index: int, start_point: Vector, end_point: Vector, heading: float, min_elevation: float, max_elevation: float):
        self.line_index = line_index
        self.point_index = point_index
        self.start_point = start_point
        self.end_point = end_point
        self.heading = heading
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation

    def to_json(self):
        return {
            'line_index': self.line_index,
            'point_index': self.point_index,
            'start_point': d_vec3(self.start_point),
            'end_point': d_vec3(self.end_point),
            'heading': self.heading,
            'min_elevation': self.min_elevation,
            'max_elevation': self.max_elevation,
        }

class LineGeometryConfig:
    pass

# SECTION // Helper functions

def d_vec3(vector):
    '''Convert a Blender vector to a JSON/dictionary'''
    return {
        'x': vector.x,
        'y': vector.y,
        'z': vector.z,
    }

def geojson_line(feature_id, points, label=""):
    '''Convert a 3D point (Vector) into GeoJSON (get rid of z coordinate)'''
    return { "type": "Feature", "properties": { "fid": feature_id, "label": label }, "geometry": { "type": "LineString", "coordinates": [ list(p)[:2] for p in points ] } }

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
    bound_box[0].resize_3d()
    bound_box[6].resize_3d()
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

# find x-y spacing of a grid DEM (in local space), as well as grid width and height
def find_spacing(object):
    x_spacing = abs(object.data.vertices[1].co.x - object.data.vertices[0].co.x)

    width = 1
    y_coord = object.data.vertices[0].co.y
    y_coord_last = object.data.vertices[width].co.y
    while abs(y_coord - y_coord_last) < 0.001:
        y_coord_last = object.data.vertices[width].co.y
        width += 1

    y_spacing = abs(object.data.vertices[width].co.y - object.data.vertices[0].co.y)
    return (width - 1, len(object.data.vertices) // width, x_spacing, y_spacing)

def advect_particle(object, start_vertex_index):
    '''
    Follow a particle's path downhill and return the indices that it travels to
    '''
    bm = bmesh.new()
    bm.from_mesh(object.data)
    bm.verts.ensure_lookup_table()

    advect_path_indices = []
    last_downhill_index = -1
    downhill_index = start_vertex_index
    while last_downhill_index != downhill_index:
        last_downhill_index = downhill_index
        start_vertex = bm.verts[downhill_index]
        # find lowest neighbor index, but never go backwards (mockup of gradient descent w/momentum)
        downhill_index = lowest_neighbor_index(start_vertex, advect_path_indices)
        # object.data.vertices[downhill_index].select = True
        advect_path_indices.append(downhill_index)

    bm.free()
    return advect_path_indices

def lowest_neighbor_index(vert: bmesh.types.BMVert, exclude_verts: list) -> int:
    downhill_z = math.inf
    downhill_index = vert.index
    for e in vert.link_edges:
        v_neighbor = e.other_vert(vert)
        if v_neighbor.co.z < downhill_z and v_neighbor.index not in exclude_verts:
            downhill_index = v_neighbor.index
            downhill_z = v_neighbor.co.z
    return downhill_index

def make_label_with_plaform(label_center: Vector, name: str, config: LineGeometryConfig) -> tuple:
    # create label platform
    bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, enter_editmode=False, align='WORLD', location=label_center, scale=(config.label_radius, config.label_radius, config.label_height))
    label_platform = bpy.context.active_object
    label_platform.name = 'label_platform_' + name

    # create text objects
    label_string = name
    bpy.ops.object.text_add(enter_editmode=False, align='WORLD', location=label_center + Vector((0, 0, -config.label_text_height * 1.1)), scale=(1, 1, 1))
    label_text = bpy.context.active_object
    label_text.data.body = label_string
    label_text.data.align_x = 'CENTER'
    label_text.data.align_y = 'CENTER'
    label_text.data.size = config.label_radius * config.label_text_size
    label_text.data.font = bpy.data.fonts['Arial Regular'] # assumes Ariel was loaded correctly at the beginning
    label_text.name = 'label_text_' + label_string
    with bpy.context.temp_override(active_object=label_text):
        bpy.ops.object.convert(target='MESH')
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, config.label_text_height * 2)})
        bpy.ops.object.mode_set(mode='OBJECT')
    return (label_platform, label_text)

'''
Generate legend ticks

Algorithm from D3.js, source code:
https://github.com/d3/d3-array/blob/main/src/ticks.js
'''

import math

e10 = math.sqrt(50)
e5 = math.sqrt(10)
e2 = math.sqrt(2)


def ticks(start: float, stop: float, count: float) -> list:
    i = 0
    ticks = []

    if (start == stop and count > 0):
        return [start]

    reverse = stop < start
    if (reverse):
        n = start
        start = stop
        stop = n

    step = tickIncrement(start, stop, count)
    if (step == 0) or not math.isfinite(step):
        return []

    if (step > 0):
        r0 = round(start / step)
        r1 = round(stop / step)

        if (r0 * step < start):
            r0 += 1
        if (r1 * step > stop):
            r1 -= 1
        n = r1 - r0 + 1
        ticks = [0 for _ in range(n)]
        while i < n:
            ticks[i] = (r0 + i) * step
            i += 1
    else:
        step = -step
        r0 = round(start * step)
        r1 = round(stop * step)

        if (r0 / step < start):
            r0 += 1
        if (r1 / step > stop):
            r1 -= 1
        n = r1 - r0 + 1
        ticks = [0 for _ in range(n)]
        while (i < n):
            ticks[i] = (r0 + i) / step
            i += 1

    if (reverse):
        ticks.reverse()

    return ticks


def tickIncrement(start: float, stop: float, count: float) -> float:
    step = (stop - start) / max(0, count)
    power = int(math.log(step) / math.log(10))
    error = step / math.pow(10, power)
    if power >= 0:
        return (10 if error >= e10 else (5 if error >= e5 else (2 if error >= e2 else 1))) * math.pow(10, power)
    else:
        return -math.pow(10, -power) / (10 if error >= e10 else (5 if error >= e5 else (2 if error >= e2 else 1)))

def tickStep(start: float, stop: float, count: float) -> float:
    step0 = abs(stop - start) / max(0, count)
    step1 = math.pow(10, int(math.log(step0) / math.log(10)))
    error = step0 / step1

    if (error >= e10):
        step1 *= 10
    elif (error >= e5):
        step1 *= 5
    elif (error >= e2):
        step1 *= 2

    return -step1 if stop < start else step1

def map_range(value, low1, high1, low2, high2):
    '''
    remap a value from range (low1, high1) => (low2, high2)
    '''
    return low2 + (high2 - low2) * (value - low1) / (high1 - low1)











# SECTION // Main Function

def main():
    ################################################################################
    # STAGE 0: Setup / configuration
    # - calculate world (blender) space coordinates, bounding boxes, etc.
    # - specify line geometry configuration

    print('Creating physicalization task geometry and answers')
    # set random seed for reproducible results
    random.seed(3)

    # load ariel font
    bpy.ops.font.open(filepath="C:/Windows/Fonts/arial.ttf", relative_path=False)

    # Get objects from Blender
    heightmap = bpy.context.active_object
    if heightmap is None:
        print('No heightmap selected')
        return

    # create bmesh and BVH tree for raycasts
    heightmap_bmesh = bmesh.new()
    heightmap_bmesh.from_mesh(heightmap.data)
    heightmap_bvh = BVHTree.FromBMesh(heightmap_bmesh)

    # faster to work with a list of custom vertices than Blender structs
    vertices = [VertexLite(v.index, v.co, v.select) for v in heightmap.data.vertices]

    # find grid spacing
    grid_width, grid_height, x_spacing, y_spacing = find_spacing(heightmap)
    print('grid width', 'grid height', 'x spacing (m)', 'y spacing (m)')
    print(grid_width, grid_height, x_spacing, y_spacing)
    x_spacing_world = x_spacing * heightmap.scale.x
    y_spacing_world = y_spacing * heightmap.scale.y

    ################################################################################
    # DEFINE CONSTANTS
    # world bounds on a 2x2x2 cube (for example):
    # (-1.0, -1.0, -1.0)
    # (-1.0, -1.0, 1.0)
    # (-1.0, 1.0, 1.0)
    # (-1.0, 1.0, -1.0)
    # (1.0, -1.0, -1.0)
    # (1.0, -1.0, 1.0)
    # (1.0, 1.0, 1.0)
    # (1.0, 1.0, -1.0)
    HEIGHTMAP_WORLD_BOUNDS = [heightmap.matrix_world @ Vector(heightmap.bound_box[i]) for i in range(8)]
    HEIGHTMAP_BOUNDS = [Vector(heightmap.bound_box[i]) for i in range(8)]
    heightmap_center = (HEIGHTMAP_WORLD_BOUNDS[6] + HEIGHTMAP_WORLD_BOUNDS[0]) / 2.0
    heightmap_size = HEIGHTMAP_WORLD_BOUNDS[6] - HEIGHTMAP_WORLD_BOUNDS[0]
    # for b in world_bounds:
    #     bpy.ops.mesh.primitive_uv_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=b, scale=(1, 1, 1))
    LINE_OBJ_NAME = 'line_'
    POINT_OBJ_NAME = 'point_'

    config = LineGeometryConfig()

    # line distribution
    config.num_lines = 10
    config.num_points = 10
    config.point_spacing = 150.0 / 11.0 # dataset size in millimeters / spaces between lines
    config.line_spacing = 150.0 / 11.0
    config.line_length = (config.num_points + 1) * config.point_spacing

    # line/point shape
    config.line_width = 0.25 #mm
    config.line_tmp_height = (HEIGHTMAP_WORLD_BOUNDS[6].z - HEIGHTMAP_WORLD_BOUNDS[0].z) * 2.0 #mm
    config.point_hash_length = 2.0 #mm

    # label constants
    config.label_radius = 3 #mm
    config.label_text_size = 2.0 # % of label radius
    config.label_height = 0.30 #mm below surface
    config.label_text_height = 0.30 #mm above label

    # answer calculation constants
    config.range_iteration_step = y_spacing_world
    config.num_advect_points = 10
    config.advect_radius = config.point_spacing
    config.major_advect_directions = 12 # cardinal direction hashes to create
    # config.minor_advect_directions = 16 # cardinal direction hashes to create

    # legend constants
    config.legend_platform_height = 1.0 #mm (overlaps with legend bars)
    config.target_legend_steps = 10
    config.legend_bar_height = 5 #mm (slightly bigger than text height)
    config.legend_bar_width = 9 #mm (slightly bigger than max text width)
    config.legend_bar_spacing = 0 #mm

    ################################################################################
    # STAGE 1: Precalculations
    # - define line segments (start and end) for grid
    # - pre-calculate all the vertices that are near every line so lookups are faster (memoization)

    # Part 0: define line start/end XY points in world space (NO elevation yet)
    line_segments_xy = []
    for l in range(config.num_lines):
        start_x = HEIGHTMAP_WORLD_BOUNDS[0].x
        end_x = HEIGHTMAP_WORLD_BOUNDS[6].x
        y = HEIGHTMAP_WORLD_BOUNDS[6].y - l * config.line_spacing - config.line_spacing
        line_segments_xy.append((Vector((start_x, y)), Vector((end_x, y))))

    # Part 1: Find vertices in the DEM near each line
    print('Memoizing vertices near each line on DEM')
    # optimization: pre-select vertices that are close to each line
    # use start and end points to create bounding box, then add vertex indices
    line_bounding_boxes_xy = []
    for i, (pt1, pt2) in enumerate(line_segments_xy):
        min_x, max_x = min(pt1.x, pt2.x), max(pt1.x, pt2.x)
        min_y, max_y = min(pt1.y, pt2.y), max(pt1.y, pt2.y)

        min_pt = Vector((min_x, min_y))
        max_pt = Vector((max_x, max_y))
        extents = (max_pt - min_pt) * 0.5
        if extents.x < config.point_hash_length:
            min_pt.x -= config.point_hash_length
            max_pt.x += config.point_hash_length
        if extents.y < config.point_hash_length:
            min_pt.y -= config.point_hash_length
            max_pt.y += config.point_hash_length

        # use grid spacing buffer to ensure we get all edge vertices too
        min_pt.x -= x_spacing_world
        max_pt.x += x_spacing_world
        min_pt.y -= y_spacing_world
        max_pt.y += y_spacing_world

        bbox = [min_pt] + [None] * 5 + [max_pt] + [None]
        line_bounding_boxes_xy.append(bbox)

    line_bbox_vertices = [[] for _ in range(len(line_segments_xy))]
    for v, vertex in enumerate(vertices):
        if v % (len(vertices) // 5) == 0:
            print('    -> Progress: {:.0%}'.format(v / len(vertices)))
        world_pos = heightmap.matrix_world @ vertex.co
        for i, bbox in enumerate(line_bounding_boxes_xy):
            if within_bounds(world_pos.resized(3), bbox, check_axes=(True, True, False)):
                line_bbox_vertices[i].append(vertex)
    print('    -> Found vertices for each line: ', [len(l) for l in line_bbox_vertices])


    ################################################################################
    # STAGE 2: Calculate Task Answers
    # 1. SORT and COMPARE task: go through each line, find its world coordinates (and DEM coordinates), and do an elevation lookup
    # 2. Range task: iterate along each line to find the min/max elevation (in DEM coordinates)
    # 3. Direction Task: find suitable points for advection

    # Part 1: Find the actual 3D coordinates of every point of every line in world space
    # can convert to DEM space later if necessary
    print('Calculating sort/compare task answers')
    lines_points = []
    for l in range(config.num_lines):
        print('    -> Line {} / {}'.format(l, config.num_lines))
        points = []
        for p in range(config.num_points):
            px = HEIGHTMAP_WORLD_BOUNDS[0].x + config.point_spacing * p + config.point_spacing
            start_pt, _end_pt = line_segments_xy[l]
            line_point = Vector((px, start_pt.y, 0))
            line_point_dem = heightmap.matrix_world.inverted() @ line_point

            # do a raycast to check height at the point
            loc, _normal, _index, _dist = heightmap_bvh.ray_cast(line_point_dem + Vector((0, 0, HEIGHTMAP_BOUNDS[6].z)), Vector((0, 0, -1)))
            loc_world = heightmap.matrix_world @ loc
            line_point.z = loc_world.z

            # find closest index for advect task later
            closest_pt = line_bbox_vertices[l][0].co
            closest_index = 0
            for v in line_bbox_vertices[l]:
                v_world = heightmap.matrix_world @ vertices[v.index].co
                if (line_point - v_world).magnitude < (line_point - closest_pt).magnitude:
                    closest_pt = v_world
                    closest_index = v.index

            line_point_dem = heightmap.matrix_world.inverted() @ line_point
            points.append(LinePoint(line_point_dem, line_point, closest_index))
        lines_points.append(points)

    # Part 2: Iterate through all lines and find the min/max elevation. Do rows and columns.
    # print('Calculating range task answers')
    # line_elevation_ranges = []
    # for l, (pt1, pt2) in enumerate(line_segments_xy):
    #     # raycast along the whole line
    #     world_start = pt1.copy()
    #     world_start.resize_3d()
    #     world_start.z = HEIGHTMAP_WORLD_BOUNDS[6].z + 1.0
    #     dem_start = heightmap.matrix_world.inverted() @ world_start
    #     world_end = pt2.copy()
    #     world_end.resize_3d()
    #     world_end.z = HEIGHTMAP_WORLD_BOUNDS[6].z + 1.0
    #     dem_end = heightmap.matrix_world.inverted() @ world_end

    #     min_elev = float('inf')
    #     max_elev = 0
    #     iter_len = (world_end - world_start).magnitude
    #     iter_step = 1.0 / iter_len # mm (how often along the line to raycast)
    #     t = 0
    #     while t < 1.0:
    #         raycast_pt = Vector.lerp(dem_start, dem_end, t)

    #         # do a raycast to check height at the point
    #         loc, _normal, _index, _dist = heightmap_bvh.ray_cast(raycast_pt, Vector((0, 0, -1)))
    #         if loc is not None:
    #             loc_world = heightmap.matrix_world @ loc

    #             if loc.z < min_elev:
    #                 min_elev = loc.z
    #             if loc.z > max_elev:
    #                 max_elev = loc.z

    #         t += iter_step

    #     line_elevation_ranges.append((min_elev, max_elev))

    # Part 3: Find the maximum height points from Part 1 to use as advection targets
    print('Calculating advection and range task answers')
    all_points = [((l, p), lines_points[l][p]) for l in range(config.num_lines) for p in range(config.num_points)]
    all_points.sort(key=lambda p: p[1].coord.z)
    advect_points = []
    advect_answers = []
    for ppoint in reversed(all_points):
        (l, p), start_point = ppoint
        # also don't allow any bottom-row or first-column targets (they would overlap with labels)
        # keep = l < config.num_lines - 1 and p > 0
        # labels are now outside
        keep = True
        # exclude if the point is too close to an existing target (no overlaps allowed)
        for (al, ap), apoint in advect_points:
            if (apoint.world_coord - start_point.world_coord).magnitude < config.advect_radius * 2:
                keep = False
                break
        # Part 4: find actual advection task answers
        # find the full path of the particle until it exits the model or fails to move
        particle_path = advect_particle(heightmap, start_point.index)

        # shorten the path to inside the advect radius
        for path_index, idx in enumerate(particle_path):
            point_world = heightmap.matrix_world @ vertices[idx].co
            if (point_world.xy - start_point.world_coord.xy).magnitude > config.advect_radius:
                break

        particle_path_short = particle_path[:path_index]
        particle_path_long = particle_path[path_index + 1:]

        # ensure the particle didn't get stuck in a valley somewhere inside the radius
        keep = keep and len(particle_path_long) > 0

        if not keep:
            continue

        # DEBUG
        # bpy.ops.mesh.primitive_uv_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=start_point.world_coord, scale=(1, 1, 1))
        # for idx in particle_path_short:
        #     vertices[idx].select = True
        # start_point.select = True
        # vertices[start_point.index].select = True

        # determine the end point on the circle
        end_point = heightmap.matrix_world @ vertices[particle_path_short[-1]].co

        # find signed start to end heading in world coords
        start_to_end = (end_point - start_point.world_coord).normalized()

        # signed angle using atan2: https://stackoverflow.com/a/33920320
        forward = Vector((0, 1, 0))
        up = Vector((0, 0, 1))
        va = forward
        vb = start_to_end
        angle = math.atan2(Vector.dot(Vector.cross(va, vb), up), Vector.dot(va, vb))

        # heading in scene rotation (ccw, -180 ~ 180 signed, north is 0)
        heading_blender = math.degrees(angle)

        # convert to unsigned clockwise 0 ~ 360 heading format (north still 0)
        heading_deg = -heading_blender
        if heading_deg < 0:
            heading_deg += 360

        # # DEBUG
        # bpy.ops.mesh.primitive_cone_add(radius1=1, radius2=0, depth=10, enter_editmode=False, align='WORLD', location=start_point.world_coord, scale=(1, 1, 1))
        # cone = bpy.context.active_object
        # c_up = Vector((0, 0, 1))
        # c_fwd = start_to_end.copy()
        # c_fwd.z = 0
        # c_right = Vector.cross(c_fwd, c_up)
        # cone_basis = Matrix([
        #     [c_right.x, c_up.x, c_fwd.x, 0],
        #     [c_right.y, c_up.y, c_fwd.y, 0],
        #     [c_right.z, c_up.z, c_fwd.z, 0],
        #     [0, 0, 0, 1],
        # ])
        # cone.matrix_world = cone.matrix_world @ cone_basis
        # cone.name = 'start_to_end'

        # # DEBUG
        # bpy.ops.mesh.primitive_cone_add(radius1=1, radius2=0, depth=5, enter_editmode=False, align='WORLD', location=start_point.world_coord, scale=(1, 1, 1))
        # cone = bpy.context.active_object
        # rotation = Euler((-math.pi / 2, 0, math.radians(heading_blender)),  'XYZ').to_matrix().to_4x4()
        # cone.matrix_world = cone.matrix_world @ rotation
        # cone.name = 'out_heading'


        # Part 5: calculate range task answers (range task now is a circle -- calculate min/max value within the circle)
        # Sample concentric circles around the surface to find the min/max elevation
        min_elev = float('inf')
        max_elev = 0
        ring_spacing = x_spacing_world
        ring_radius = 0
        while ring_radius < config.advect_radius:
            t = 0
            while t <= math.pi * 2:
                x = ring_radius * math.cos(t) + start_point.world_coord.x
                y = ring_radius * math.sin(t) + start_point.world_coord.y
                z = HEIGHTMAP_WORLD_BOUNDS[6].z + 1.0

                raycast_pt_world = Vector((x, y, z))
                raycast_pt = heightmap.matrix_world.inverted() @ raycast_pt_world
                # bpy.ops.mesh.primitive_cone_add(radius1=1, radius2=0, depth=5, enter_editmode=False, align='WORLD', location=raycast_pt, scale=(1, 1, 1))

                # do a raycast to check height at the point (in object space)
                loc, _normal, _index, _dist = heightmap_bvh.ray_cast(raycast_pt, Vector((0, 0, -1)))

                if loc is not None:
                    loc_world = heightmap.matrix_world @ loc

                    if loc.z < min_elev:
                        # bpy.ops.mesh.primitive_uv_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=loc_world, scale=(1, 1, 1))
                        min_elev = loc.z
                    if loc.z > max_elev:
                        # bpy.ops.mesh.primitive_cone_add(radius1=1, radius2=0, depth=5, enter_editmode=False, align='WORLD', location=loc_world, scale=(1, 1, 1))
                        max_elev = loc.z
                t += math.pi / 24
            ring_radius += ring_spacing

        advect_answers.append(AdvectAnswer(l, p, start_point.world_coord, end_point, heading_deg, min_elev, max_elev))
        advect_points.append(ppoint)

        if len(advect_answers) >= config.num_advect_points:
            break

    # propagate any selections made back to og geometry
    for i, v in enumerate(vertices):
        heightmap.data.vertices[i].select = v.select


    ################################################################################
    # STAGE 3: Generate geometry
    # - horizontal lines
    # - vertical ticks (points)
    # - circles at advection task points
    # - line labels at left of rows
    # - point labels at top of columns
    #
    # all geometry is made very tall so that it can be INTERSECT'd with the main DEM later
    # note: point geometry uses the closest vertex, NOT the original(ideal) XYZ value

    if MAKE_GEOMETRY:
        print('Generating overlay geometry')

        # Part 1-2: Generate "crosshair" markings at each point
        print('    -> Crosshairs')
        crosshair_offset = Vector((0.75 * config.point_hash_length, 0, 0))
        crosshair_scale = Vector((config.point_hash_length, config.line_width, config.line_tmp_height))
        point_geometry = []
        for l, line in enumerate(lines_points):
            for p, start_point in enumerate(line):
                # east part of crosshair
                bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=start_point.world_coord + crosshair_offset, scale=crosshair_scale)
                pt = bpy.context.active_object
                pt.name = POINT_OBJ_NAME + index_to_letter(l) + str(p + 1)
                point_geometry.append(pt)

                # west part of crosshair
                bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=start_point.world_coord - crosshair_offset, scale=crosshair_scale)
                pt = bpy.context.active_object
                pt.name = POINT_OBJ_NAME + index_to_letter(l) + str(p + 1)
                point_geometry.append(pt)

                # north part of crosshair
                bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=start_point.world_coord + crosshair_offset.yxz, scale=crosshair_scale.yxz)
                pt = bpy.context.active_object
                pt.name = POINT_OBJ_NAME + index_to_letter(l) + str(p + 1)
                point_geometry.append(pt)

                # south part of crosshair
                bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=start_point.world_coord - crosshair_offset.yxz, scale=crosshair_scale.yxz)
                pt = bpy.context.active_object
                pt.name = POINT_OBJ_NAME + index_to_letter(l) + str(p + 1)
                point_geometry.append(pt)
        with bpy.context.temp_override(active_object=point_geometry[0], selected_editable_objects=point_geometry):
            bpy.ops.object.join()

        # Part 3: Generate advection circles
        print('    -> Advection circles')
        circle_geometries = []
        for (l, p), start_point in advect_points:
            # inner circle
            bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, enter_editmode=False, align='WORLD', location=start_point.world_coord, scale=(config.advect_radius - config.line_width, config.advect_radius - config.line_width, config.line_tmp_height))
            inner_circle = bpy.context.active_object
            # outer circle
            bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=1, enter_editmode=False, align='WORLD', location=start_point.world_coord, scale=(config.advect_radius, config.advect_radius, config.line_tmp_height))
            circle_geom = bpy.context.active_object

            circle_geom.name = 'advect_circles'
            bool_mod = circle_geom.modifiers.new(name='diff', type='BOOLEAN')
            bool_mod.operation = 'DIFFERENCE'
            bool_mod.object = inner_circle
            with bpy.context.temp_override(active_object=circle_geom):
                for modifier in circle_geom.modifiers:
                    bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.context.collection.objects.unlink(inner_circle)
            circle_geometries.append(circle_geom)

        with bpy.context.temp_override(active_object=circle_geometries[0], selected_editable_objects=circle_geometries):
            bpy.ops.object.join()


        # Part 3.5: Generate advection circle minor cardinal direction ticks
        print('    -> Advection circle hash marks')
        cardinal_direction_geom = []
        for i, ((l, p), start_point) in enumerate(advect_points):
            print('        ->', i, '/', len(advect_points))
            # major cardinal ticks
            cardinal_north = start_point.world_coord + Vector((0, config.advect_radius - config.point_hash_length * 0.5 - config.line_width, 0))
            bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=cardinal_north, scale=(config.line_width, config.point_hash_length, config.line_tmp_height))
            cardinal = bpy.context.active_object
            cardinal.name = 'cardinal_directions_' + index_to_letter(l) + str(p + 1)
            with bpy.context.temp_override(active_object=cardinal):
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.spin(steps=config.major_advect_directions, angle=math.pi * 2, center=start_point.world_coord, axis=(0, 0, 1))
                bpy.ops.object.mode_set(mode='OBJECT')
            cardinal_direction_geom.append(cardinal)

            # minor cardinal ticks
            # cardinal_north = point.world_coord + Vector((0, config.advect_radius - config.point_hash_length * 0.2, 0))
            # bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=cardinal_north, scale=(config.line_width, config.point_hash_length * 0.25, config.line_tmp_height))
            # cardinal = bpy.context.active_object
            # cardinal.name = 'cardinal_directions_' + index_to_letter(l) + str(p + 1)
            # with bpy.context.temp_override(active_object=cardinal):
            #     bpy.ops.object.mode_set(mode='EDIT')
            #     bpy.ops.mesh.spin(steps=config.minor_advect_directions, angle=math.pi * 2, center=point.world_coord, axis=(0, 0, 1))
            #     bpy.ops.object.mode_set(mode='OBJECT')
            # cardinal_direction_geom.append(cardinal)

        with bpy.context.temp_override(active_object=cardinal_direction_geom[0], selected_editable_objects=cardinal_direction_geom):
            bpy.ops.object.join()


        # Part 4: Join all overlay geometry (lines/point hashes/circles)
        print('    -> Joining all overlay geometry...')
        points_geom = point_geometry[0]
        points_geom.name = 'crosshairs'
        cardinal_direction_geom = cardinal_direction_geom[0]

        circles_union = points_geom.modifiers.new(name='union circles', type='BOOLEAN')
        circles_union.operation = 'UNION'
        circles_union.object = circle_geometries[0]

        cardinals_union = points_geom.modifiers.new(name='union cardinals', type='BOOLEAN')
        cardinals_union.operation = 'UNION'
        cardinals_union.object = cardinal_direction_geom

        with bpy.context.temp_override(active_object=points_geom):
            for modifier in points_geom.modifiers:
                bpy.ops.object.modifier_apply(modifier=modifier.name)

        bpy.context.collection.objects.unlink(circle_geometries[0])
        bpy.context.collection.objects.unlink(cardinal_direction_geom)
        print('Created overlay geometry')

    if MAKE_LABELS:
        print('Generating labels')
        # Part 5: Generate labels for lines (rows)
        label_platforms = []
        label_texts = []
        for l, (start, end) in enumerate(line_segments_xy):
            # center label to the left of all crosshairs
            left_center = Vector((
                start.x - config.label_radius,
                start.y,
                config.legend_platform_height - config.label_height * 0.9
            ))
            right_center = Vector((
                end.x + config.label_radius,
                end.y,
                config.legend_platform_height - config.label_height * 0.9
            ))
            label_platform, label_text = make_label_with_plaform(left_center, index_to_letter(l), config)
            label_platforms.append(label_platform)
            label_texts.append(label_text)
            label_platform, label_text = make_label_with_plaform(right_center, index_to_letter(l), config)
            label_platforms.append(label_platform)
            label_texts.append(label_text)

        # Part 6: Make column (point hash) labels
        # Assume all columns are equal across the entire height
        for p, start_point in enumerate(lines_points[0]):
            # center below all crosshairs
            below_center = Vector((
                start_point.world_coord.x,
                HEIGHTMAP_WORLD_BOUNDS[0].y - config.label_radius,
                config.legend_platform_height - config.label_height * 0.9
            ))
            above_center = Vector((
                start_point.world_coord.x,
                HEIGHTMAP_WORLD_BOUNDS[6].y + config.label_radius,
                config.legend_platform_height - config.label_height * 0.9
            ))
            label_platform, label_text = make_label_with_plaform(below_center, str(p + 1), config)
            label_platforms.append(label_platform)
            label_texts.append(label_text)
            label_platform, label_text = make_label_with_plaform(above_center, str(p + 1), config)
            label_platforms.append(label_platform)
            label_texts.append(label_text)

        # join individual labels
        with bpy.context.temp_override(active_object=label_platforms[0], selected_editable_objects=label_platforms):
            bpy.ops.object.join()
        with bpy.context.temp_override(active_object=label_texts[0], selected_editable_objects=label_texts):
            bpy.ops.object.join()

        # Part 7: Make flat bases for labels to go on
        label_bases = []

        base_width = config.label_radius
        left_pos = Vector((HEIGHTMAP_WORLD_BOUNDS[0].x - base_width, heightmap_center.y, config.legend_platform_height / 2.0))
        right_pos = Vector((HEIGHTMAP_WORLD_BOUNDS[6].x + base_width, heightmap_center.y, config.legend_platform_height / 2.0))
        bot_pos = Vector((heightmap_center.x, HEIGHTMAP_WORLD_BOUNDS[0].y - base_width, config.legend_platform_height / 2.0))
        top_pos = Vector((heightmap_center.x, HEIGHTMAP_WORLD_BOUNDS[6].y + base_width, config.legend_platform_height / 2.0))

        # make them overlap a bit with the DEM model
        vert_size = Vector((base_width * 2.1, heightmap_size.y + base_width * 4, config.legend_platform_height))
        horiz_size = Vector((heightmap_size.x + base_width * 4, base_width * 2.1, config.legend_platform_height))

        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=left_pos, scale=vert_size)
        label_bases.append(bpy.context.active_object)
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=right_pos, scale=vert_size)
        label_bases.append(bpy.context.active_object)
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=top_pos, scale=horiz_size)
        label_bases.append(bpy.context.active_object)
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=bot_pos, scale=horiz_size)
        label_bases.append(bpy.context.active_object)

        # join label bases (union because they intersect)
        for u in range(1, len(label_bases)):
            label_base_union = label_bases[0].modifiers.new(name='union labels', type='BOOLEAN')
            label_base_union.operation = 'UNION'
            label_base_union.object = label_bases[u]

        with bpy.context.temp_override(active_object=label_bases[0]):
            bpy.context.view_layer.objects.active = label_bases[0]
            for modifier in label_bases[0].modifiers:
                bpy.ops.object.modifier_apply(modifier=modifier.name)

            for u in range(1, len(label_bases)):
                bpy.context.collection.objects.unlink(label_bases[u])

        label_platform_geom = label_platforms[0]
        label_text_geom = label_texts[0]
        label_base_geom = label_bases[0]
        label_platform_geom.name = 'labels_platforms'
        label_text_geom.name = 'labels_text'
        label_base_geom.name = 'labels_base'

        # join point labels with text (union because they intersect), but don't apply in case they need to be separate
        # label_platform_union = label_base_geom.modifiers.new(name='union plaforms', type='BOOLEAN')
        # label_platform_union.operation = 'UNION'
        # label_platform_union.object = label_platform_geom

        # label_text_union = label_base_geom.modifiers.new(name='union text', type='BOOLEAN')
        # label_text_union.operation = 'UNION'
        # label_text_union.object = label_text_geom
        # label_point_union = label_point_platform_geom.modifiers.new(name='union labels', type='BOOLEAN')
        # label_point_union.operation = 'UNION'
        # label_point_union.object = label_point_text_geom

        print('Created labels')

    ################################################################################
    # STAGE 4: Create the legend
    # legend sea level (0m) starts at PLATFORM, not at TABLE level

    # initialize heights and ticks
    min_dem_height = min(vertices, key=lambda v: v.co.z).co.z
    max_dem_height = max(vertices, key=lambda v: v.co.z).co.z
    min_world_height = heightmap.scale.z * min_dem_height
    max_world_height = heightmap.scale.z * max_dem_height
    tick_entries = ticks(min_dem_height, max_dem_height, config.target_legend_steps)
    tick_entries.reverse()
    config.legend_width = (config.legend_bar_height + config.legend_bar_spacing) * len(tick_entries)

    # if MAKE_LEGEND:
    #     # make legend platform
    #     bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=Vector((0, 0, config.legend_platform_height/2)), scale=(config.legend_width, config.legend_bar_width, config.legend_platform_height))
    #     legend_platform_object = bpy.context.active_object
    #     legend_platform_object.name = heightmap.name + '_legend'

    # create legend bars and labels
    legend_entries = []
    legend_objects = []
    legend_upper_text = []
    legend_side_text = []
    bar_index = 0
    for bar_index in range(0, len(tick_entries)):
        x_co = -config.legend_width / 2 + bar_index * (config.legend_bar_height + config.legend_bar_spacing) + (config.legend_bar_height + config.legend_bar_spacing) / 2
        dem_height = tick_entries[bar_index]
        # height EXCLUDES platform height (platform is sea level.)
        world_height = map_range(dem_height, min_dem_height, max_dem_height, min_world_height, max_world_height)
        adjusted_height = world_height - config.legend_platform_height
        bar_position = Vector((x_co, 0, adjusted_height / 2))
        bar_scale = Vector((config.legend_bar_height, config.legend_bar_width, adjusted_height))
        legend_entries.append(LegendEntry(dem_height, bar_scale, bar_position))

        if MAKE_LEGEND:
            bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=bar_position, scale=bar_scale)
            bar_object = bpy.context.active_object
            bar_object.name = heightmap.name + '_legend_bar'
            legend_objects.append(bar_object)

            # make text to sit on top of the bars
            label_pos = Vector((bar_position.x, bar_position.y, adjusted_height - config.label_text_height))
            bpy.ops.object.text_add(enter_editmode=False, align='WORLD', location=label_pos, scale=(1, 1, 1))
            label_text = bpy.context.active_object
            label_text.data.body = '{:.0f}'.format(dem_height)
            label_text.data.align_x = 'CENTER'
            label_text.data.align_y = 'CENTER'
            label_text.data.size = config.legend_bar_height
            label_text.data.font = bpy.data.fonts['Arial Regular'] # assumes Ariel was loaded correctly at the beginning
            label_text.name = heightmap.name + '_legend_text_' + str(bar_index)
            label_text.rotation_euler = Euler((0, 0, math.pi / 2), 'XYZ')
            with bpy.context.temp_override(active_object=label_text):
                bpy.ops.object.convert(target='MESH')
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, config.label_text_height * 2.0)})
                bpy.ops.object.mode_set(mode='OBJECT')
            legend_upper_text.append(label_text)

            # make text for "close" side (-y) of bars. don't render if too small
            label_pos = Vector((bar_position.x, bar_position.y - config.legend_bar_width / 2, 0))
            bpy.ops.object.text_add(enter_editmode=False, align='WORLD', location=label_pos, scale=(1, 1, 1))
            label_text = bpy.context.active_object
            label_text.data.body = '{:.0f}'.format(dem_height)
            label_text.data.align_x = 'LEFT'
            label_text.data.align_y = 'CENTER'
            label_text.data.size = config.legend_bar_height
            label_text.data.font = bpy.data.fonts['Arial Regular'] # assumes Ariel was loaded correctly at the beginning
            label_text.name = heightmap.name + '_legend_text_' + str(bar_index)
            label_text.rotation_euler = Euler((0, -math.pi / 2, math.pi / 2), 'XYZ')

            # squish if too small
            char_width = config.legend_bar_height / 2.0
            text_width = len(label_text.data.body) * char_width
            if text_width > adjusted_height:
                label_text.data.body = label_text.data.body.replace('m', '')
                text_width = len(label_text.data.body) * char_width
            with bpy.context.temp_override(active_object=label_text):
                bpy.ops.object.convert(target='MESH')
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, -config.label_text_height, 0)})
                bpy.ops.object.mode_set(mode='OBJECT')

            if text_width > adjusted_height:
                label_text.scale.x = adjusted_height / text_width
            legend_side_text.append(label_text)

            # make text for "far" side (+y) of bars. don't render if too small
            label_pos = Vector((bar_position.x, bar_position.y + config.legend_bar_width / 2, 0))
            bpy.ops.object.text_add(enter_editmode=False, align='WORLD', location=label_pos, scale=(1, 1, 1))
            label_text = bpy.context.active_object
            label_text.data.body = '{:.0f}'.format(dem_height)
            label_text.data.align_x = 'LEFT'
            label_text.data.align_y = 'CENTER'
            label_text.data.size = config.legend_bar_height
            label_text.data.font = bpy.data.fonts['Arial Regular'] # assumes Ariel was loaded correctly at the beginning
            label_text.name = heightmap.name + '_legend_text_' + str(bar_index)
            label_text.rotation_euler = Euler((0, -math.pi / 2, -math.pi / 2), 'XYZ')

            # squish if too small
            text_width = len(label_text.data.body) * char_width
            if text_width > adjusted_height:
                label_text.data.body = label_text.data.body.replace('m', '')
                text_width = len(label_text.data.body) * char_width
            with bpy.context.temp_override(active_object=label_text):
                bpy.ops.object.convert(target='MESH')
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, config.label_text_height, 0)})
                bpy.ops.object.mode_set(mode='OBJECT')

            if text_width > adjusted_height:
                label_text.scale.x = adjusted_height / text_width
            legend_side_text.append(label_text)

            bar_index += 1

    if MAKE_LEGEND:
        # with bpy.context.temp_override(active_object=legend_objects[0], selected_editable_objects=legend_objects):
        #     bpy.ops.object.join()
        with bpy.context.temp_override(active_object=legend_objects[0]):
            for o in range(1, len(legend_objects)):
                name = 'union ' + legend_objects[o].name
                legend_objects[0].modifiers.new(name=name, type='BOOLEAN')
                legend_objects[0].modifiers[name].operation = 'UNION'
                legend_objects[0].modifiers[name].object = legend_objects[o]

            bpy.context.view_layer.objects.active = legend_objects[0]
            for modifier in legend_objects[0].modifiers:
                bpy.ops.object.modifier_apply(modifier=modifier.name, report=True)
            legend_objects[0] = heightmap.name + '_legend_bar'

        for o in range(1, len(legend_objects)):
            bpy.data.objects.remove(legend_objects[o], do_unlink=True)

        with bpy.context.temp_override(active_object=legend_upper_text[0], selected_editable_objects=legend_upper_text):
            bpy.ops.object.join()
        with bpy.context.temp_override(active_object=legend_side_text[0], selected_editable_objects=legend_side_text):
            bpy.ops.object.join()

        # with bpy.context.temp_override(active_object=legend_platform_object):
        #     legend_platform_object.modifiers.new(name='Boolean', type='BOOLEAN')
        #     legend_platform_object.modifiers["Boolean"].operation = 'UNION'
        #     legend_platform_object.modifiers["Boolean"].object = bpy.data.objects['legend_bar']
        #     bpy.ops.object.modifier_apply(modifier="Boolean", report=True)
        #     bpy.data.objects.remove(bpy.data.objects['legend_bar'], do_unlink=True)
        print('Created legend geometry')


    ################################################################################
    # STAGE 5: Calculate real-world coordinates for each point so it can be used in a geojson

    # geo coordinates of origin for geojson
    bgis = importlib.import_module("BlenderGIS-228")
    geo_scene = bgis.geoscene.GeoScene(bpy.context.scene)
    config.crsx = geo_scene.crsx
    config.crsy = geo_scene.crsy

    line_geojson = []
    rotate90 = Euler((0, 0, math.pi / 2), 'XYZ')
    hash_scale = (0.5 * config.point_hash_length) / heightmap.scale.x

    # convert back to DEM space and get index too
    coord_ref_xform = Matrix([
        [1, 0, 0, config.crsx],
        [0, 1, 0, config.crsy],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    crs_xform = coord_ref_xform @ heightmap.matrix_world.inverted()

    # calculate extents of data in EPSG:3857 coordinates
    geojson_extents = [
        (config.crsx + HEIGHTMAP_BOUNDS[0].x, config.crsy + HEIGHTMAP_BOUNDS[0].y),
        (config.crsx + HEIGHTMAP_BOUNDS[6].x, config.crsy + HEIGHTMAP_BOUNDS[6].y),
    ]
    (minx, miny), (maxx, maxy) = geojson_extents
    geojson_extents_str = ','.join([str(s) for s in [minx, maxx, miny, maxy]]) + ' [EPSG:3857]'

    fid = 0
    for l, points in enumerate(lines_points):
        print('    -> Finding GeoJSON geometry for line {} / {}'.format(l, len(lines_points)))
        line_points_meters_crs = [crs_xform @ points[p].world_coord for p in range(len(points))]
        crosshair_offset = Vector((0.75 * config.point_hash_length, 0, 0))
        crosshair_scale = Vector((config.point_hash_length, config.line_width, config.line_tmp_height))

        # add crosshairs
        for p, pt in enumerate(points):
            # east crosshair
            start_pt = pt.world_coord + crosshair_offset / 2
            end_pt = pt.world_coord + crosshair_offset / 2 + Vector((config.point_hash_length, 0, 0))
            start_pt = crs_xform @ start_pt
            end_pt = crs_xform @ end_pt
            linestring = geojson_line(fid, [start_pt, end_pt])
            fid += 1
            line_geojson.append(linestring)

            # west crosshair
            start_pt = pt.world_coord - crosshair_offset / 2
            end_pt = pt.world_coord - crosshair_offset / 2 - Vector((config.point_hash_length, 0, 0))
            start_pt = crs_xform @ start_pt
            end_pt = crs_xform @ end_pt
            linestring = geojson_line(fid, [start_pt, end_pt])
            fid += 1
            line_geojson.append(linestring)

            # south crosshair
            start_pt = pt.world_coord - crosshair_offset.yxz / 2
            end_pt = pt.world_coord - crosshair_offset.yxz / 2 - Vector((0, config.point_hash_length, 0))
            start_pt = crs_xform @ start_pt
            end_pt = crs_xform @ end_pt
            linestring = geojson_line(fid, [start_pt, end_pt])
            fid += 1
            line_geojson.append(linestring)

            # north crosshair
            start_pt = pt.world_coord + crosshair_offset.yxz / 2
            end_pt = pt.world_coord + crosshair_offset.yxz / 2 + Vector((0, config.point_hash_length, 0))
            start_pt = crs_xform @ start_pt
            end_pt = crs_xform @ end_pt
            linestring = geojson_line(fid, [start_pt, end_pt])
            fid += 1
            line_geojson.append(linestring)

        # add some spoof lines at the bottom to get point labels
        if l == config.num_lines - 1:
            for p, start_point in enumerate(lines_points[0]):
                label_center = Vector((
                    start_point.world_coord.x,
                    HEIGHTMAP_WORLD_BOUNDS[0].y - config.label_radius * 1.5,
                ))
                label_center.resize_3d()
                label_center = crs_xform @ label_center
                hash_x = Vector((hash_scale * 3, 0, 0))
                linestring = geojson_line(fid, [label_center - hash_x, label_center + hash_x], str(p + 1))
                fid += 1
                line_geojson.append(linestring)

        if l == 0:
            for p, start_point in enumerate(lines_points[0]):
                label_center = Vector((
                    start_point.world_coord.x,
                    HEIGHTMAP_WORLD_BOUNDS[6].y + config.label_radius * 1.5,
                ))
                label_center.resize_3d()
                label_center = crs_xform @ label_center
                hash_x = Vector((hash_scale * 3, 0, 0))
                linestring = geojson_line(fid, [label_center - hash_x, label_center + hash_x], str(p + 1))
                fid += 1
                line_geojson.append(linestring)

        for l, points in enumerate(lines_points):
            start_point = points[0]
            label_center = Vector((
                HEIGHTMAP_WORLD_BOUNDS[0].x - config.label_radius * 1.5,
                start_point.world_coord.y,
            ))
            label_center.resize_3d()
            label_center = crs_xform @ label_center
            hash_x = Vector((hash_scale * 3, 0, 0))
            linestring = geojson_line(fid, [label_center - hash_x, label_center + hash_x], index_to_letter(l))
            fid += 1
            line_geojson.append(linestring)

            end_point = points[-1]
            label_center = Vector((
                HEIGHTMAP_WORLD_BOUNDS[6].x + config.label_radius * 1.5,
                end_point.world_coord.y,
            ))
            label_center.resize_3d()
            label_center = crs_xform @ label_center
            hash_x = Vector((hash_scale * 3, 0, 0))
            linestring = geojson_line(fid, [label_center - hash_x, label_center + hash_x], index_to_letter(l))
            fid += 1
            line_geojson.append(linestring)

    # add circles for gradients
    for (l, p), start_point in advect_points:
        circle_center = crs_xform @ start_point.world_coord
        circle_radius = config.advect_radius / heightmap.scale.x
        circle_cardinal_len = (config.point_hash_length) / heightmap.scale.x
        circle_minor_cardinal_len = (config.point_hash_length * 0.25) / heightmap.scale.x

        circle_pts = []
        step = 0.1
        t = 0
        while t < math.pi * 2:
            x = math.cos(t) * circle_radius
            y = math.sin(t) * circle_radius
            pt = Vector((x, y, 0)) + circle_center
            circle_pts.append(pt)
            t += step
        linestring = geojson_line(fid, circle_pts)
        line_geojson.append(linestring)
        fid += 1

        # add major cardinal directions to circles
        step = math.pi * 2 / config.major_advect_directions
        t = 0
        while t < math.pi * 2:
            x1 = math.cos(t) * circle_radius
            y1 = math.sin(t) * circle_radius
            x2 = math.cos(t) * (circle_radius - circle_cardinal_len)
            y2 = math.sin(t) * (circle_radius - circle_cardinal_len)
            linestring = geojson_line(fid, [circle_center + Vector((x1, y1, 0)), circle_center + Vector((x2, y2, 0))])
            fid += 1
            line_geojson.append(linestring)
            t += step

        # add minor cardinal directions to circles
        # step = math.pi * 2 / config.minor_advect_directions
        # t = 0
        # while t < math.pi * 2:
        #     x1 = math.cos(t) * circle_radius
        #     y1 = math.sin(t) * circle_radius
        #     x2 = math.cos(t) * (circle_radius - circle_minor_cardinal_len)
        #     y2 = math.sin(t) * (circle_radius - circle_minor_cardinal_len)
        #     linestring = geojson_line(fid, [circle_center + Vector((x1, y1, 0)), circle_center + Vector((x2, y2, 0))], 'cardinal_' + index_to_letter(l) + str(p + 1))
        #     fid += 1
        #     line_geojson.append(linestring)
        #     t += step

    ################################################################################
    # STAGE 5: Export the answers to files

    if MAKE_FILE_OUTPUT:
        export_path = PATH.joinpath('LineStudySources/' + heightmap.name).expanduser()
        if not export_path.exists():
            os.makedirs(export_path)
        export_file = export_path.joinpath('metadata.json')

        # export JSON of answers
        print('FINISHED finding task answers.')
        with open(export_file, 'w') as fout:
            out_json = {
                'scale': d_vec3(heightmap.scale),
                'demBounds': [d_vec3(v) for v in HEIGHTMAP_BOUNDS],
                'extents': geojson_extents,
                'extentsString': geojson_extents_str,
                'config': config.__dict__,
                'legendEntries': [e.__dict__ for e in legend_entries],
                'demElevationRange': (min_dem_height, max_dem_height),
                # 'lineElevationRanges': line_elevation_ranges,
                'linesPoints': [[p.to_json() for p in line] for line in lines_points],
                'advectionPoints': [p.to_json() for p in advect_answers],
            }
            json.dump(out_json, fout, indent=4)
        print('Exported answers to JSON', export_file)

        # save a geojson for all lines/hashmarks
        geojson = {
            "type": "FeatureCollection",
            "name": heightmap.name + "_lines",
            "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3857" } },
            "features": line_geojson
        }
        with open(str(export_file).replace('json', 'geojson'), 'w') as fout:
            json.dump(geojson, fout, indent=4)
            print('Exported lines to geojson')


if __name__ == '__main__':
    main()