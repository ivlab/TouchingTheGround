# Data from the user study

- coding-responses.xlsx: raw questionnaire responses from participants with cells removed that had identifiable information
- confidence.csv: confidence, reported at the end of each section (modality + task pairing) -- 9 entries per participant
- participants-anonymized.csv: demographic information
- questionnaire-normalized.csv: normalized questionnaire after coding
- trials-raw.csv: raw trial data from the experiment (timing and actual user responses)
- trials-raw_preprocessed.csv: raw trial data preprocessed to calculate error from user responses and total time (calculated with `analysis/preprocess_responses.py`)