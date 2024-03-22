from main import PROCESSED_DATA_FILENAME, RESAMPLED_DATA_FILENAME

RUN_DIFF = True
STEP = 60
if STEP == 15:
    DATA_FILENAME = PROCESSED_DATA_FILENAME
else:
    DATA_FILENAME = RESAMPLED_DATA_FILENAME

if RUN_DIFF:
    prefix = "diff_"
else:
    prefix = ""

suffix = str(STEP) + "m"

LABEL_NAME = "cert_" + suffix
MEASURED_NAME = "SCADA_" + suffix
FORECASTER_NAME = "latest_" + suffix

MAX_EPOCHS = 400
BATCH_SIZE = 128
LAGS = 5
# TODO for lags 3, 5, 7, 9, 20
OUT_STEPS = 1

