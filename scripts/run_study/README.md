# Run the study sever

## Folder contents

- `generate_task_ordering.py`: Set up balanced Latin squares for Task and Modality and pre-generate the ordering. Outputs a CSV that is used to enter data in the `study_data` folder.
- `parse_study_answers.py`: Parse user responses from a spreadsheet and calculate error. Used by the `preprocess_responses.py` in the `analysis` folder.
- `run_study_server.py`: Actually run the study. Serves the webpage for participants to advance the study using the tablet.


## Prerequisites

Install Python Flask.

```
python3 -m pip install flask
```


## Running the server

```
python3 run_study_server.py <participant_id>
```

Then, access from a browser at http://localhost:5000 OR http://<this computer's
IP address>:5000.