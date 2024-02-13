# destructively moves a dataset vertically so that its lowest point is at a particular height.
# makes a copy of each object in the collection, then shortens each object in the collection
import bpy
from mathutils import Vector
import importlib
from pathlib import Path
import json

OUTPUT_MIN_HEIGHT = 1 #mm above floor, in world coordinates

def main():
    collection = bpy.context.collection
    new_collection = bpy.data.collections.new('Shortened')
    collection.children.link(new_collection)

    bgis = importlib.import_module("BlenderGIS-228")
    geo_scene = bgis.geoscene.GeoScene(bpy.context.scene)
    crsx = geo_scene.crsx
    crsy = geo_scene.crsy

    differences = {}
    for obj in collection.objects:
        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        min_height_vert = min(new_obj.data.vertices, key=lambda v: v.co.z)
        diff = min_height_vert.co.z - OUTPUT_MIN_HEIGHT / new_obj.scale.z
        new_obj.name += '_short_' + str(diff)

        for i in range(len(new_obj.data.vertices)):
            new_obj.data.vertices[i].co.z -= diff
        new_collection.objects.link(new_obj)

        # calculate extents of data in EPSG:3857 coordinates
        bbox = [Vector(b) for b in obj.bound_box]
        geojson_extents = [
            (crsx + bbox[0].x, crsy + bbox[0].y),
            (crsx + bbox[6].x, crsy + bbox[6].y),
        ]
        (minx, miny), (maxx, maxy) = geojson_extents
        geojson_extents_str = ','.join([str(s) for s in [minx, maxx, miny, maxy]]) + ' [EPSG:3857]'

        differences[obj.name] = {
            'input_min_height': min(obj.data.vertices, key=lambda v: v.co.z).co.z,
            'output_min_height': OUTPUT_MIN_HEIGHT / new_obj.scale.z,
            'height_difference': diff,
            'extent': geojson_extents_str
        }

    save_path = Path('~/Documents/research/proposal/LineStudySources/{}/inter/{}'.format('mt_whitney', 'shortened.json')).expanduser()
    if not save_path.parent.exists():
        save_path.parent.mkdir()

    with open(save_path, 'w') as fout:
        json.dump(differences, fout, indent=4)

if __name__ == '__main__':
    main()