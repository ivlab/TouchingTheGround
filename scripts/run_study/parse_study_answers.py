import json
import sys
from pathlib import Path
import re

DATASET_PATH = Path('../../datasets').expanduser()
METADATA_FILENAME = 'vis/metadata.json'

def heading_to_clock(heading: float) -> str:
    clock = heading / 30.0
    if clock < 1.0:
        clock += 12.0

    hours = int(clock)
    minutes = round((clock - hours) * 60)
    return '{}:{:02.0f}'.format(hours, minutes)

def clock_to_heading(clock_numeral: str) -> float:
    hours, minutes = [int(c) for c in clock_numeral.split(':')]
    minutes_dec = minutes / 60
    clock = (hours + minutes_dec) % 12.0
    return clock * 30.0

def range_answer(metadata_json: dict, range_target: str):
    num_re = re.compile(r'(\w)(\d+)')
    re_match = num_re.match(range_target)
    line_index = ord(re_match.group(1)) - 65
    point_index = int(re_match.group(2)) - 1
    advection_points = metadata_json['advectionPoints']
    try:
        pt = next(filter(lambda p: p['line_index'] == line_index and p['point_index'] == point_index, advection_points))
        return ['{:.0f}'.format(p) for p in [pt['min_elevation'], pt['max_elevation']]]
    except StopIteration:
        return None

def sort_answer(metadata_json: dict, sort_line: str):
    line_index = ord(sort_line.lower()) - 97
    line_points = metadata_json['linesPoints'][line_index]
    line_points_sorted = [(i, p) for i, p in enumerate(line_points)]
    line_points_sorted.sort(key=lambda p: p[1]['coord']['z']) # ensure *really* sorted
    return [str(i + 1) for i, _point in line_points_sorted]

def compare_answer(metadata_json: dict, pt1: str, pt2: str, pt3: str):
    pts = [pt1, pt2, pt3]
    num_re = re.compile(r'([A-Za-z]+)([0-9]+)')
    elevations = []
    for pt in pts:
        re_match = num_re.match(pt)
        pt_letter = re_match.group(1)
        line_index = ord(pt_letter.lower()) - 97
        line_points = metadata_json['linesPoints'][line_index]
        pt_index = int(re_match.group(2))
        point = line_points[pt_index - 1]
        elevations.append(point['coord']['z'])
    min_index, _min_elevation = min(enumerate(elevations), key=lambda ie: ie[1])
    return [pts[min_index]]

def advect_answer(metadata_json: dict, point: str):
    num_re = re.compile(r'(\w)(\d+)')
    re_match = num_re.match(point)
    line_index = ord(re_match.group(1)) - 65
    point_index = int(re_match.group(2)) - 1
    advection_points = metadata_json['advectionPoints']
    try:
        pt = next(filter(lambda p: p['line_index'] == line_index and p['point_index'] == point_index, advection_points))
        heading = pt['heading']
        return heading_to_clock(heading)
    except StopIteration:
        return None


def get_metadata_json(dataset_name: str):
    metadata_path = DATASET_PATH.joinpath(dataset_name, METADATA_FILENAME)
    metadata_json = None
    with open(metadata_path) as fin:
        metadata_json = json.load(fin)
    return metadata_json

def main():

    if len(sys.argv) != 4:
        helps = '''
usage: python3 parse_study_answers.py <dataset_name> <task name> <target>
where:
    - dataset_name: name of the dataset/geometry folder where metadata.json can be found
    - task name: name of the task (range / compare / advect)
    - target: the target
'''
        print(helps)
        return 1

    dataset_name, task_name, target = sys.argv[1:]

    metadata_json = get_metadata_json(dataset_name)

    task_answer_fn = ANSWERS[task_name]
    print('`{}` task answer for target `{}` with dataset `{}`: '.format(task_name, target, dataset_name), task_answer_fn(metadata_json, target))


ANSWERS = {
    'range': range_answer,
    'sort': sort_answer,
    'compare': compare_answer,
    'advect': advect_answer,
}


if __name__ == '__main__':
   exit(main())