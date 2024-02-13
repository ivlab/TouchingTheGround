import os
import time
import sys
from flask import Flask, render_template, Response
from pathlib import Path
import generate_task_ordering
import threading
import socket
import signal
import json
import queue

app = Flask(__name__)

DATA_OUT_PATH = Path('~/Documents/research/proposal/LineStudyData').expanduser()
PARTICIPANT_ID = -1
TRIALS = []
STATE = {
    'current_trial': 0,
    'started_trial': -1,
    'trial_start_times': [],
    'trial_end_times': [],
}
SOCKET_OPEN = threading.Event()
SOCKET_SERVER = None
MESSAGE_QUEUE = None
SHUTDOWN_EVENT = "PhysStudyShutdown"
RESET_EVENT = "PhysStudyReset"

@app.route('/')
def index():
    return render_template('index.html', participant_id=PARTICIPANT_ID)

@app.route('/api/trial')
def trial_data():
    current_trial = STATE['current_trial']
    if current_trial < len(TRIALS):
        return TRIALS[current_trial].__dict__
    else:
        return {}

@app.route('/api/get-total-trials')
def get_total_trials():
    return {
        'numTrials': len(TRIALS),
        'trialsPerBlock': len(TRIALS) // len(generate_task_ordering.MODALITIES)
    }

@app.route('/api/start-trial', methods=['POST'])
def start_trial():
    if STATE['started_trial'] == -1:
        STATE['trial_start_times'].append(time.time())
        STATE['started_trial'] = STATE['current_trial']
        return Response(status=200)
    else:
        return Response('Trial already started', status=400)

@app.route('/api/reset-unity', methods=['GET'])
def reset_unity():
    # tell Unity to reset
    MESSAGE_QUEUE.append({'m_Name': RESET_EVENT, 'm_DataTypeName': ''})
    return Response(status=200)


@app.route('/api/end-trial', methods=['POST'])
def end_trial():
    if STATE['started_trial'] != STATE['current_trial'] - 1:
        return Response('Trial mismatch', status=400)

    STATE['trial_end_times'].append(time.time())
    with open(DATA_OUT_PATH.joinpath(str(PARTICIPANT_ID) + '_trial_times.csv'), 'w') as fout:
        rows = zip(STATE['trial_start_times'], STATE['trial_end_times'])
        rows_out = ['start_time,end_time\n']
        rows_out += [','.join([str(t) for t in r]) + '\n' for r in rows]
        fout.writelines(rows_out)
        print('WROTE TRIAL DATA (end of trial)', STATE['current_trial'])

    STATE['started_trial'] = -1

    # tell Unity to reset
    MESSAGE_QUEUE.append({'m_Name': RESET_EVENT, 'm_DataTypeName': ''})

    if STATE['current_trial'] >= len(TRIALS):
        return Response('END OF TRIALS', status=200)
    else:
        return Response(status=200)

@app.route('/api/next-trial', methods=['POST'])
def next_trial(increment: int=1):
    if STATE['started_trial'] != STATE['current_trial']:
        return Response('Trial mismatch', status=400)

    current_trial = STATE['current_trial']
    # STATE['current_trial'] = min(len(TRIALS) - 1, max(0, STATE['current_trial']))
    print()
    # print('CURRENT TRIAL:')
    if STATE['current_trial'] + 1 < len(TRIALS):
        ntrial = current_trial + 1
        print('{} / {}  ({})'.format(ntrial + 1, len(TRIALS), TRIALS[ntrial].task))
        print(TRIALS[ntrial].dataset, TRIALS[ntrial].modality, sep=" - ")
    else:
        print('NO MORE TRIALS')
    print()
    if increment > 0:
        STATE['current_trial'] += 1
    else:
        STATE['current_trial'] -= 1
    return Response(status=200)

@app.route('/api/restart', methods=['POST'])
def restart():
    STATE['current_trial'] = 0
    print('RESTARTED')
    return Response(status=200)

@app.route('/api/get-question-text')
def get_question_text():
    return generate_task_ordering.QUESTION_TEXT

@app.route('/api/pause-modal')
def pause_modal():
    print('*' * 80)
    print('*' * 80)
    print()
    input('Press <Enter> when ready for participant to continue: ')
    print()
    print('*' * 80)
    print('*' * 80)
    return Response(status=200)

def run_socket_server():
    global MESSAGE_QUEUE
    MESSAGE_QUEUE = []
    socket.setdefaulttimeout(1.0)
    SOCKET_SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SOCKET_SERVER.bind(('0.0.0.0', 5001))
    SOCKET_SERVER.listen()
    SOCKET_OPEN.set()
    # print(sys.byteorder)

    while SOCKET_OPEN.is_set():
        # accept ONE connection
        connected = False
        while not connected:
            # print('waiting to connect')
            try:
                (connection, address) = SOCKET_SERVER.accept()
                connected = True
            except TimeoutError:
                pass
        print('accepted connection from', address)

        while connected and SOCKET_OPEN.is_set():
            # sender
            while len(MESSAGE_QUEUE) > 0:
                msg = MESSAGE_QUEUE.pop(0)
                msg_json = json.dumps(msg)
                msg_len_json = len(msg_json)

                # send length
                length_bytes = int.to_bytes(msg_len_json, 4, 'little')
                connection.send(length_bytes)

                # send message
                msg_bytes = msg_json.encode(encoding='ascii')
                connection.send(msg_bytes)


            # receiver
            # receive message length int
            try:
                msglen = -1
                msglen_bytes = connection.recv(4)
                msglen = int.from_bytes(msglen_bytes, byteorder='little')
            except TimeoutError:
                pass
            
            if msglen < 0:
                continue

            # receive message
            chunks = []
            bytes_recd = 0
            while bytes_recd < msglen and connected and SOCKET_OPEN.is_set():
                try:
                    chunk = connection.recv(min(msglen - bytes_recd, 2049))
                    chunks.append(chunk)
                    bytes_recd = bytes_recd + len(chunk)
                except TimeoutError:
                    pass

            msg = b''.join(chunks).decode(encoding='ascii')
            try:
                msg_json = json.loads(msg)
                if msg_json['m_Name'] == SHUTDOWN_EVENT:
                    print('Unity client disconnected')
                    connected = False
            except:
                pass

    print('socket server shutting down')
    SOCKET_SERVER.close()

def main():
    if len(sys.argv) < 2:
        print('usage: python3 generate_task_ordering.py <participant_id_0_indexed> <?trial num to startt at>')
        return 1

    use_dataset = None
    # if len(sys.argv) == 3:
    #     use_dataset = sys.argv[2]
    if not DATA_OUT_PATH.exists():
        os.makedirs(DATA_OUT_PATH)

    global PARTICIPANT_ID, TRIALS
    PARTICIPANT_ID = int(sys.argv[1])
    TRIALS = generate_task_ordering.generate_task_ordering(PARTICIPANT_ID, use_dataset)

    if len(sys.argv) == 3:
        current_trial = int(sys.argv[2]) - 1
    else:
        current_trial = 0

    STATE['current_trial'] = current_trial
    print()
    print('{} / {}  ({})'.format(current_trial + 1, len(TRIALS), TRIALS[current_trial].task))
    print(TRIALS[current_trial].dataset, TRIALS[current_trial].modality, sep=" - ")
    print()

    socket_server_thread = threading.Thread(target=run_socket_server)
    socket_server_thread.daemon = True
    socket_server_thread.start()

    app.run(host="0.0.0.0", debug=False)
    print('flask app shutting down')

    SOCKET_OPEN.clear()
    socket_server_thread.join(1)

if __name__ == '__main__':
    exit(main())