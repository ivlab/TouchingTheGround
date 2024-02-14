# Data from the user study

- confidence.csv: confidence, reported at the end of each section (modality + task pairing) -- 9 entries per participant
- participants-anonymized.csv: demographic information
- questionnaire-anonymized.csv: questionnaire results with cells removed that had identifiable information
- trials-raw.csv: raw trial data from the experiment (timing and actual user responses)
- trials-raw_preprocessed.csv: raw trial data preprocessed to calculate error from user responses and total time (calculated with `analysis/preprocess_responses.py`)