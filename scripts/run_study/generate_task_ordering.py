import random
import sys
import parse_study_answers

# = p participants × 3 Modalities × 3 Datasets × 3 Tasks × 3 Repetitions
# 
# - 3 modalities (balanced latin square ordering) [MAIN FACTOR. we expect there to be an effect of modality on time & errors performance here.]
#   - 3 tasks (balanced latin square ordering) [Secondary Factor. Comparison between tasks is mostly irrelevant.]
#       - 1 dataset (balanced latin square ordering BETWEEN subjects.)
#
# Overall, we expect to see an effect of Modality on Task Time and Error Rate, as well as Confidence.
# We may see interaction effects between modality and task.
# We DO NOT expect to see order effects.
#
# Better:
# Since we have 3 modalities, we need multiples of 6 participants (2 * 3) for
# balanced latin square ordering to work correctly.
# 
# GPower a priori analysis tells us (if we use ONE dataset):
# F tests - ANOVA: Repeated measures, within factors
# Analysis:	A priori: Compute required sample size 
# Input:	Effect size f	=	.25
# 	α err prob	=	0.05
# 	Power (1-β err prob)	=	0.95
# 	Number of groups	=	3
# 	Number of measurements	=	3
# 	Corr among rep measures	=	0.5
# 	Nonsphericity correction ε	=	1
# Output:	Noncentrality parameter λ	=	16.8750000
# 	Critical F	=	3.1051566
# 	Numerator df	=	2.0000000
# 	Denominator df	=	84.0000000
# 	Total sample size	=	45
# 	Actual power	=	0.9597015
#
# Interpreting the results for ONE dataset, this means we need > 15 participants
# (45 total samples / 3 groups) to end up with an effect size 0.25 (medium).
#
# If we use TWO datasets:
# F tests - ANOVA: Repeated measures, within factors
# Analysis:	A priori: Compute required sample size 
# Input:	Effect size f	=	.25
# 	α err prob	=	0.05
# 	Power (1-β err prob)	=	0.95
# 	Number of groups	=	3
# 	Number of measurements	=	6
# 	Corr among rep measures	=	0.5
# 	Nonsphericity correction ε	=	1
# Output:	Noncentrality parameter λ	=	22.5000000
# 	Critical F	=	2.2813047
# 	Numerator df	=	5.0000000
# 	Denominator df	=	135
# 	Total sample size	=	30
# 	Actual power	=	0.9665574
#
# Interpreting the results for TWO datasets, this means we need > 10 participants
# (30 total samples / 3 groups) to end up with an effect size 0.25 (medium).
#
#
# Pending results of more pilot studies (e.g., if we have time to do two
# datasets), we could use 12 participants (TWO datasets) or 18 participants (ONE
# dataset). Could have more too... depending on effect size.
#
# Old news:
# Since we have 3 modalities, we need multiples of 6 participants (2 * 3) for
# balanced latin square ordering to work correctly. So, according to a priori
# power analysis, 30 participants is good to shoot for (3 groups, 6 measurements in GPower).
#
# All assuming an effect size of 0.25.

index_to_letter = lambda i: chr(65 + i)

# print(participant_id, modality, dataset, task, rep, '-'.join(target), '-'.join(answer), sep=',')
class Trial:
    def __init__(self, participant_id, trial_num, modality, dataset, task, rep, target, answer, modality_order, dataset_order, task_order):
        self.participant_id = participant_id
        self.trial_num = trial_num
        self.modality = modality
        self.dataset = dataset
        self.task = task
        self.rep = rep
        self.target = target
        self.answer = answer

        self.modality_order = modality_order
        self.dataset_order = dataset_order
        self.task_order = task_order

    def __str__(self):
        return ",".join([str(self.participant_id), str(self.trial_num), self.modality, self.task, self.dataset, str(self.modality_order), str(self.task_order), str(self.dataset_order), str(self.rep), self.target, self.answer])

MODALITIES = ['physical', '2d', '3d']
TRAINING_DATASET = '0-0'
DATASETS = ['0-2', '0-1', '2-0']
# TASKS = ['range', 'sort', 'compare', 'direction']
TASKS = ['range', 'compare', 'advect']
REPETITIONS = 3
TRAINING_REPETITIONS = 1
QUESTION_TEXT = {
    'range': 'Indicate the range of elevations inside the circle surrounding __target__.',
    'sort': 'On __target__, sort the numbered points by their elevation, ascending.',
    'compare': 'Locate the three given points and determine which point has the lowest elevation.',
    # 'advect': 'If a small ball were released at __target__, what heading (o\'clock) on the circle would it end up at?'
    'advect': 'Trace the path downhill from __target__ until the path intersects the surrounding circle. What heading (o\'clock) does the path intersect the circle?'
}

# from generate_line_study_geometry
num_lines = 10
num_points = 10

# https://gist.github.com/graup/70b09323bfa7182fe693eecb8e749896#file-balanced_latin_squares-py
def balanced_latin_squares(n):
    l = [[((j//2+1 if j%2 else n-j//2) + i) % n for j in range(n)] for i in range(n)]
    if n % 2:  # Repeat reversed for odd n
        l += [seq[::-1] for seq in l]
    return l

# Get lines in an orderly manner. Try to minimize rememberance between tasks too
# (offset the latin square by a block)
# due to new grid design, ensure we don't get lines right next to each other in a row.
def get_lines(pid, modality, task):
    lsq = balanced_latin_squares(num_lines)
    lines_participant = lsq[abs(pid) % len(lsq)]
    lines = [chr(65 + l) for l in lines_participant]

    modality_index = MODALITIES.index(modality)
    task_index = TASKS.index(task)
    max_index = len(MODALITIES) * REPETITIONS

    # offset by modality and task index
    # offset = ((modality_index + task_index) * REPETITIONS) % max_index
    # result = lines[offset : offset + REPETITIONS]
    # ensure we don't end up with lines next to each other
    offset = (modality_index + task_index) % REPETITIONS
    result = []
    i = offset
    for rep in range(REPETITIONS):
        result.append(lines[i])
        i = (i + REPETITIONS) % len(lines)


    # shuffles based on participant ID only since random.seed was set - so for
    # every participant this result is the same.
    random.shuffle(result)

    return result

def generate_line():
    return chr(random.randint(65, 65 + num_lines - 1))

def generate_point():
    return generate_line() + str(random.randint(1, num_points))

def generate_unique_points(n_points):
    '''Generate points on unique lines (don't have two points on the same line ever)'''
    pts = []
    i = 0
    while i < n_points:
        pt = generate_point()
        found_match = True
        for p in pts:
            if pt[0] == p[0]:
                found_match = False
        if found_match:
            pts.append(pt)
            i += 1
    return pts

def generate_distant_points(dataset_name, n_points, radius):
    '''Generate unique points that are outside a particular radius from each other'''
    metadata = parse_study_answers.get_metadata_json(dataset_name)
    points_world_coords = [pt['world_coord'] for line in metadata['linesPoints'] for pt in line]
    
    pts = []
    iis = []
    i = 0
    while i < n_points:
        new_i = random.randint(0, len(points_world_coords) - 1)
        new_p = points_world_coords[new_i]
        found_match = True
        # ensure new points are outside radius from all existing points
        for p in pts:
            if (p['x'] - new_p['x']) ** 2 + (p['y'] - new_p['y']) ** 2 + (p['z'] - new_p['z']) ** 2 < radius ** 2:
                found_match = False
                break

        if found_match:
            pts.append(new_p)
            iis.append(new_i)
            i += 1

    # convert indices to row/column
    final_points = []
    for ii in iis:
        row = ii // num_lines
        col = ii % num_lines
        pt = index_to_letter(row) + str(col + 1)
        final_points.append(pt)
    
    return final_points

def generate_advect_point(metadata_json, already_chosen):
    advection_points = metadata_json['advectionPoints']
    points = [index_to_letter(e['line_index']) + str(e['point_index'] + 1) for e in advection_points if (index_to_letter(e['line_index']) + str(e['point_index'] + 1)) not in already_chosen]
    return random.choice(points)

def invalid_points(metadata_json, buffer=10):
    '''return all the points that are "invalid":
    - too close to the edge of the model (within `buffer` millimeters from edge)
    '''
    model_scale = metadata_json['scale']
    buffer = (1.0 / model_scale['x']) * buffer #mm, in real-world space, transformed into DEM-space
    min_x = metadata_json['demBounds'][0]['x']
    max_x = metadata_json['demBounds'][6]['x']
    min_y = metadata_json['demBounds'][0]['y']
    max_y = metadata_json['demBounds'][6]['y']

    bad_points = []
    for l, pts in enumerate(metadata_json['linesPoints']):
        line_letter = chr(l + 65)
        for pi, pt in enumerate(pts):
            coords = pt['coord']
            cx = coords['x']
            cy = coords['y']
            cz = coords['z']
            diffs = []
            diffs.append(abs(min_x - cx))
            diffs.append(abs(max_x - cx))
            diffs.append(abs(min_y - cy))
            diffs.append(abs(max_y - cy))

            if min(diffs) < buffer:
                bad_points.append(line_letter + str(pi + 1))
    return bad_points

def generate_task_ordering(participant_id: int, use_only_dataset: str):
    # ensure reproducible results
    random.seed(participant_id)

    # balanced latin square for modality type (between participants)
    balanced_latin_squares_result = balanced_latin_squares(len(MODALITIES))
    modality_order = balanced_latin_squares_result[abs(participant_id) % len(balanced_latin_squares_result)]
    modalities = [MODALITIES[i] for i in modality_order]

    # balanced latin square for task presentation order (between modalities within participants)
    lsq_tasks = balanced_latin_squares(len(TASKS))

    # Generate order for datasets (ensure datasets end up with equal number of data points)
    # abandon 7/20
    # dataset_orders = []
    # for d in range(len(DATASETS)):
    #     dataset_orders.append([d, (d + 1) % len(DATASETS)])
    # dataset_list = [DATASETS[d] for d in dataset_orders[abs(participant_id) % len(DATASETS)]]
    #
    # instead, use a single dataset (just give 'em one based on the participant ID)
    dataset_list = [DATASETS[abs(participant_id) % len(DATASETS)]]
    print('Participant ', participant_id, 'gets datasets', dataset_list)

    # dataset_list = DATASETS
    tasks = TASKS

    if use_only_dataset:
        dataset_list = []
        global TRAINING_DATASET
        TRAINING_DATASET = use_only_dataset

    # 3 Modalities (balanced latin square between subj)
    #   x 3 Tasks (balanced latin square between modalities)
    #       -> 2 training repetitions
    #       -> 6 timed repetitions (2 datasets, chosen from pool of 3, presented in random order)

    # list of targets used in the trials - try to have no duplicate entries for
    # a particular dataset across the entire experiment
    targets_dataset_task = {ds + task: [] for ds in [TRAINING_DATASET] + dataset_list for task in tasks}
    max_tries_no_duplicates = 100

    trial_num = 1
    modality_num = 0
    trials = []
    for modality in modalities:
        task_order = lsq_tasks[modality_num % len(lsq_tasks)]
        for task_num in task_order:
            task = TASKS[task_num]
            lines = get_lines(participant_id, modality, task)
            random.shuffle(dataset_list)

            dataset_num = 0
            for dataset in [TRAINING_DATASET] + dataset_list:
                metadata_json = parse_study_answers.get_metadata_json(dataset)
                # invalids = invalid_points(metadata_json, buffer=10)
                invalids = []

                # make training tasks
                if dataset == TRAINING_DATASET:
                    for rep in range(1, TRAINING_REPETITIONS + 1):
                        # keep going till we've found
                        keep = False
                        tries = 0
                        while not keep and tries < max_tries_no_duplicates:
                            if task == 'range':
                                # target = [lines[rep - 1]]
                                target = [generate_advect_point(metadata_json, targets_dataset_task[dataset + task])]
                                answer = parse_study_answers.ANSWERS['range'](metadata_json, *target)
                                keep = target[0] not in targets_dataset_task[dataset + task]
                            elif task == 'sort':
                                target = [lines[rep - 1]]
                                answer = parse_study_answers.ANSWERS['sort'](metadata_json, *target)
                                keep = target[0] not in targets_dataset_task[dataset + task]
                            elif task == 'compare':
                                target = generate_distant_points(dataset, 3, 50) #3 points 5cm away
                                answer = parse_study_answers.ANSWERS['compare'](metadata_json, *target)
                                keep = not any([t in targets_dataset_task[dataset + task] for t in target]) and not any([t in invalids for t in target])
                            elif task == 'advect':
                                target = [generate_advect_point(metadata_json, targets_dataset_task[dataset + task])]
                                answer = [parse_study_answers.ANSWERS['advect'](metadata_json, *target)]
                                keep = target[0] not in targets_dataset_task[dataset + task] and target[0] not in invalids
                            tries += 1
                        targets_dataset_task[dataset + task].extend(target)
                        trials.append(Trial(participant_id, trial_num, modality, dataset, task, 'training' + str(rep), '-'.join(target), '-'.join(answer), modality_num, dataset_num, task_num))
                        trial_num += 1
                else:
                    # make real tasks
                    for rep in range(1, REPETITIONS + 1):
                        keep = False
                        while not keep:
                            if task == 'range':
                                # target = [lines[rep - 1]]
                                target = [generate_advect_point(metadata_json, targets_dataset_task[dataset + task])]
                                answer = parse_study_answers.ANSWERS['range'](metadata_json, *target)
                                keep = target[0] not in targets_dataset_task[dataset + task]
                            elif task == 'sort':
                                target = [lines[rep - 1]]
                                answer = parse_study_answers.ANSWERS['sort'](metadata_json, *target)
                                keep = target[0] not in targets_dataset_task[dataset + task]
                            elif task == 'compare':
                                target = generate_unique_points(3)
                                target = generate_distant_points(dataset, 3, 50) #3 points 5cm away
                                answer = parse_study_answers.ANSWERS['compare'](metadata_json, *target)
                                keep = not any([t in targets_dataset_task[dataset + task] for t in target]) and not any([t in invalids for t in target])
                            elif task == 'advect':
                                target = [generate_advect_point(metadata_json, targets_dataset_task[dataset + task])]
                                answer = [parse_study_answers.ANSWERS['advect'](metadata_json, *target)]
                                keep = target[0] not in targets_dataset_task[dataset + task] and target[0] not in invalids
                        targets_dataset_task[dataset + task].extend(target)
                        trials.append(Trial(participant_id, trial_num, modality, dataset, task, rep, '-'.join(target), '-'.join(answer), modality_num, dataset_num, task_num))
                        trial_num += 1
                dataset_num += 1
            # trials.append(Trial(participant_id, trial_num, modality, dataset, task, -1, '', ''))
        modality_num += 1

    return trials

def main():
    if len(sys.argv) < 2:
        print('usage: python3 generate_task_ordering.py <participant_id_0_indexed>')
        print('    use -<participant_id> for training round')
        return 1

    use_dataset = None
    if len(sys.argv) == 3:
        use_dataset = sys.argv[2]

    participant_id = int(sys.argv[1])
    trials = generate_task_ordering(participant_id, use_dataset)
    print('Trial answers:')
    for t in trials:
        print(t)

    print()
    print('Confidence answers:')
    for i in range(1, len(trials)):
        prev_trial = trials[i - 1]
        cur_trial = trials[i]
        if prev_trial.task != cur_trial.task:
            if prev_trial.modality == cur_trial.modality:
                print(participant_id, prev_trial.modality, prev_trial.task, sep=',')
            else:
                print(participant_id, prev_trial.modality, prev_trial.task, 'questionnaire!', sep=',')

    print(participant_id, prev_trial.modality, prev_trial.task, sep=',')

    # check that new line strategy counts line up
    # sort_letter = {chr(l): 0 for l in range(65, 75)}
    # range_letter = {chr(l): 0 for l in range(65, 75)}
    # for pid in range(1000, 1020):
    #     trials = generate_task_ordering(pid, use_dataset)
    #     sort_trials = [t for t in trials if t.task == 'sort']
    #     range_trials = [t for t in trials if t.task == 'range']

    #     for l in range(65, 75):
    #         letter = chr(l)
    #         sl = sum([1 if letter == t.target else 0 for t in sort_trials])
    #         rl = sum([1 if letter == t.target else 0 for t in range_trials])
    #         sort_letter[letter] += sl
    #         range_letter[letter] += rl
    # print(sort_letter, range_letter)
    # they do, for numbers of participants % 10 == 0


if __name__ == '__main__':
    exit(main())