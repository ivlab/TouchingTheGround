import json
from enum import Enum
import sys
from pathlib import Path
import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import scipy

sys.path.append('../scripts/run_study')
import parse_study_answers
from parse_study_answers import clock_to_heading

char_to_index = lambda c: ord(c) - 65

RESPONSE_CSV = Path('../study-data/trials-raw.csv').expanduser()
DATASET_PATH = Path('../datasets').expanduser()
METADATA_JSON = 'vis/metadata.json'
METADATA = {}

class InterpretAnswer(Enum):
    Range = 0
    Sort = 1
    Compare = 2
    Advect = 3

def interpret_answer(answer: str, interpretation: InterpretAnswer):
    if interpretation == InterpretAnswer.Range:
        return [float(e) for e in answer.split('-')]
    elif interpretation == InterpretAnswer.Sort:
        return [int(e) for e in answer.split('-')]
    elif interpretation == InterpretAnswer.Compare:
        return answer
    elif interpretation == InterpretAnswer.Advect:
        return answer
    else:
        raise NotImplementedError()

def calculate_range_error(df_row: pd.Series) -> float:
    user_min, user_max = df_row['_UserDataRange']
    correct_min, correct_max = df_row['_CorrectDataRange']
    full_min, full_max = METADATA[df_row['Dataset']]['demElevationRange']
    min_diff = abs(user_min - correct_min)
    max_diff = abs(user_max - correct_max)
    avg_diff = (min_diff + max_diff) / 2.0
    return avg_diff / (full_max - full_min)

def calculate_sort_error(df_row: pd.Series) -> float:
    user_ranks = interpret_answer(df_row['UserAnswer'], InterpretAnswer.Sort)
    correct_ranks = interpret_answer(df_row['CorrectAnswer'], InterpretAnswer.Sort)
    tau, p = scipy.stats.kendalltau(user_ranks, correct_ranks)
    return 1.0 - ((tau + 1.0) / 2.0)

def calculate_direction_error(df_row: pd.Series) -> float:
    user_direction = df_row['UserAnswer']
    correct_direction = df_row['CorrectAnswer']
    difference = abs(clock_to_heading(user_direction) - clock_to_heading(correct_direction)) # degrees
    return difference / 360


def load_metadata_jsons(dataset_names):
    for n in dataset_names:
        try:
            with open(DATASET_PATH.joinpath(n).joinpath(METADATA_JSON)) as fin:
                METADATA[n] = json.load(fin)
        except:
            print('Unable to load metadata for ', n)
    return METADATA


def analyze_participant_data(df: pd.DataFrame):
    '''
    generate derived columns in df:
        - TimeOnTask: UserEndTime - UserStartTime (in seconds)
        - RangeError: average absolute difference between UserAnswer and CorrectAnswer divided by ACTUAL data range (interpret *Answer columns as "MIN-MAX" ranges)
        - SortError: normalized Kendall-Tau distance between UserAnswer and CorrectAnswer (interpret *Answer columns as series of ordered points "1-2-3-4-5")
        - CompareError: 0 if CorrectAnswer == UserAnswer, 1 if if CorrectAnswer != UserAnswer
    '''
    # load metadata jsons
    dataset_names = df['Dataset'].unique()
    metadata = load_metadata_jsons(dataset_names)

    # Calculate TimeOnTask
    df['TimeOnTask'] = df['UserEndTime'] - df['UserStartTime']

    # Calculate CompareError
    df.loc[df['Task'] == 'compare', 'CompareError'] = df['UserAnswer'].eq(df['CorrectAnswer']).map({True: 0, False: 1})

    # Calculate RangeError
    df['_UserDataRange'] = df.loc[df['Task'] == 'range', 'UserAnswer'].map(lambda a: interpret_answer(a, InterpretAnswer.Range), na_action='ignore')
    df['_CorrectDataRange'] = df.loc[df['Task'] == 'range', 'CorrectAnswer'].map(lambda a: interpret_answer(a, InterpretAnswer.Range), na_action='ignore')
    df.loc[df['Task'] == 'range', 'RangeError'] = df[df['Task'] == 'range'].apply(calculate_range_error, axis=1)

    # anything that's in a circle, add a CircleDataRange column
    def circle_data_range_helper(dataset_name, target):
        metadata_json = parse_study_answers.get_metadata_json(dataset_name)
        task_answer_fn = parse_study_answers.ANSWERS['range']
        minm, maxm = task_answer_fn(metadata_json, target)
        return float(maxm) - float(minm)

    circle_data_ranges = df.loc[(df['Task'] == 'range') | (df['Task'] == 'advect'), ('Dataset', 'Target')].apply(lambda t: circle_data_range_helper(*t), axis=1)
    df['CircleDataRange'] = circle_data_ranges
    # print(list(circle_data_ranges))

    # Calculate SortError
    # df.loc[df['Task'] == 'sort', 'SortError'] = df[df['Task'] == 'sort'].apply(calculate_sort_error, axis=1)

    # Calculate DirectionError
    df.loc[df['Task'] == 'advect', 'AdvectError'] = df[df['Task'] == 'advect'].apply(calculate_direction_error, axis=1)

    # put all errors into one column
    # df.loc[df['Task'] == 'advect', 'AdvectError'] = df[df['Task'] == 'advect'].apply(calculate_direction_error, axis=1)
    df['Error'] = df['AdvectError'].fillna(df['RangeError']).fillna(df['CompareError'])

def plot_modalities(df: pd.DataFrame):
    modalities = df['Modality'].unique()
    datasets = df['Dataset'].unique()
    modalities.sort()

    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(10)
    fig.set_figheight(6)

    for i, task in enumerate(['Range', 'Sort', 'Compare']):
        for d in datasets:
            axes[0, i].scatter([df.loc[(df['Modality'] == m) & (df['Dataset'] == d), task + 'Error'].mean() for m in modalities], modalities, label=d)

        axes[0, i].plot([df.loc[(df['Modality'] == m), task + 'Error'].mean() for m in modalities], modalities)
        axes[0, i].set_title(task + ' Task Error')
        axes[0, i].set_xlim(0, 1)

        for d in datasets:
            axes[1, i].scatter([df.loc[(df['Modality'] == m) & (df['Dataset'] == d), 'TimeOnTask'].where(df['Task'] == task.lower()).mean() for m in modalities], modalities, label=d)
        axes[1, i].plot([df.loc[(df['Modality'] == m), 'TimeOnTask'].where(df['Task'] == task.lower()).mean() for m in modalities], modalities)
        axes[1, i].set_title(task + ' Task Time')

    axes[0, 0].legend()
    axes[1, 0].legend()
    # fig.legend(datasets, bbox_to_anchor=(1.1, 1), loc='upper right')#, mode='expand', ncols=len(datasets))
    # fig.legend(datasets)#, mode='expand', ncols=len(datasets))

# one figure summary like Jansen et al. 2013
def plot_jansen(df: pd.DataFrame):
    modalities = df['Modality'].unique()
    modalities.sort()

    fig, axes = plt.subplots(2, 1)
    fig.set_figwidth(5)
    fig.set_figheight(5)
    colors = ['#C8D2FC', '#F49A9A', '#000000', '#35AB99']

    # perturb points vertically slightly so error bars don't overlap
    task_vert_offset = 0.10

    # z score for 95% CI
    z = 1.960
    num_participants = len(df['ParticipantID'].unique())
    modality_nums = np.array([i for i in range(len(modalities))])

    for i, task in enumerate(['Range', 'Compare', 'Advect']):
        time_means = [df.loc[(df['Modality'] == m), 'TimeOnTask'].where(df['Task'] == task.lower()).mean() for m in modalities]
        time_stds = [df.loc[(df['Modality'] == m), 'TimeOnTask'].where(df['Task'] == task.lower()).std() for m in modalities]
        time_ci95s = [time_means[i] + z * (time_stds[i] / (num_participants ** 0.5)) for i in range(len(modalities))]
        modality_adjusted = modality_nums + i * task_vert_offset
        axes[0].scatter(time_means, modality_adjusted, c=colors[i], label=task)
        axes[0].plot(time_means, modality_adjusted, c=colors[i])
        # axes[0].errorbar(time_means, modality_adjusted, xerr=time_ci95s, fmt='none', ecolor=colors[i], alpha=0.5)
        axes[0].set_yticks(modality_nums, labels=modalities)

        error_means = [df.loc[(df['Modality'] == m), task + 'Error'].mean() for m in modalities]
        error_stds = [df.loc[(df['Modality'] == m), task + 'Error'].std() for m in modalities]
        error_ci95s = [error_means[i] + z * (error_stds[i] / (num_participants ** 0.5)) for i in range(len(modalities))]
        axes[1].scatter(error_means, modality_adjusted, c=colors[i], label=task)
        axes[1].plot(error_means, modality_adjusted, c=colors[i])
        # axes[1].errorbar(error_means, modality_adjusted, xerr=error_ci95s, fmt='none', ecolor=colors[i], alpha=0.5)
        axes[1].set_yticks(modality_nums, labels=modalities)
        # axes.set_title(task + ' Task Time')
        axes[0].set_xlabel('time on task (s)')
        axes[0].set_ylabel('modality')
        axes[0].legend()
        axes[1].set_xlabel('normalized error')
        axes[1].set_ylabel('modality')
        axes[1].legend()

    # axes[0, 0].legend()
    # axes[1, 0].legend()
    # fig.legend(datasets, bbox_to_anchor=(1.1, 1), loc='upper right')#, mode='expand', ncols=len(datasets))
    # fig.legend(datasets)#, mode='expand', ncols=len(datasets))

def anova(df):
    modalities = df['Modality'].unique()
    modalities.sort()
    # test anova assumptions

    datasets = [(df.loc[(df['Modality'] == m), 'TimeOnTask']) for m in modalities]

    # test normality of data (shapiro-wilk)
    for m in modalities:
        print(scipy.stats.shapiro(df.loc[(df['Modality'] == m), 'TimeOnTask']))


    # f, pvalue = scipy.stats.f_oneway(*datasets)
    # print(f, pvalue)




def main():
    try:
        participant_id = int(sys.argv[1])
    except:
        participant_id = -1
    df = pd.read_csv(RESPONSE_CSV)
    if participant_id >= 0:
        df = df[df['ParticipantID'] == participant_id].copy()

    # filter out training / invalid rounds
    df = df[df['TrialNum'].notna()].copy()
    df = df[~df['Repetition'].str.contains('training')].copy()
    print(df)

    analyze_participant_data(df)
    df.to_csv(str(RESPONSE_CSV).replace('.csv', '_preprocessed.csv'))
    # df.to_clipboard(True, ',')

    # print out start/end times for each participant
    partipants = df['ParticipantID'].unique()
    print('id', 'start time\t\t', 'end time\t\t', 'duration', 'num trials', sep='\t')
    for p in partipants:
        # times = df.loc[df['ParticipantID'] == p].copy()
        # times['ST'] = times['UserStartTime'].apply(lambda t: datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(t)))
        # times['ET'] = times['UserEndTime'].apply(lambda t: datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(t)))
        # times.to_clipboard(True, ',')

        start_time = df.loc[df['ParticipantID'] == p, 'UserStartTime'].min()
        end_time = df.loc[df['ParticipantID'] == p, 'UserEndTime'].max()
        start = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(start_time))
        end = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(end_time))
        print(p, start.astimezone(pytz.timezone('US/Central')), end.astimezone(pytz.timezone('US/Central')), end - start, '     ', len(df[df['ParticipantID'] == p]), sep='\t')

    # anova(df)

    # print(df.loc[df['ParticipantID'] == participant_id, ('TrialNum', 'RangeError')])
    # print(df.loc[df['Task'] == 'sort', ('Modality', 'Dataset', 'UserAnswer', 'CorrectAnswer', 'SortError')])
    # print(df[(df['ParticipantID'] == participant_id) & (df['Dataset'] == 'mt_whitney')])
    # plot_modalities(df)
    # if participant_id > 0:
    #     # plot_modalities(df[(df['ParticipantID'] == participant_id)])
    #     plot_jansen(df[(df['ParticipantID'] == participant_id)])
    # else:
        # plot_modalities(df)
    # plot_jansen(df)
    # plot_modalities(df[(df['ParticipantID'] == participant_id) & (df['Dataset'] == 'mt_whitney')])
    # plot_modalities(df[(df['ParticipantID'] == participant_id) & (df['Dataset'] == 'wheeler_peak')])
    # plot_modalities(df[(df['ParticipantID'] == participant_id) & (df['Dataset'] == 'hoosier_hill')])
    # plot_modalities(df[df['ParticipantID'] == participant_id] & df[df['Dataset'] == 'mt_whitney'])
    # plot_datasets(df[df['ParticipantID'] == participant_id] & df[df['Dataset'] == 'mt_whitney'])
    # plt.show()



if __name__ == '__main__':
    exit(main())