
from sre_parse import CATEGORIES


TRANSFOMERS_FEATURES = ['sin_day', 'cos_day', 'sin_month', 'cos_month', 'sin_hour', 'cos_hour', 'sin_min', 'cos_min', 'Total']
CATEGORICAL_FEATURES = ["day", "month", "hour", "minute"]
OUTPUT_FEATURE = "Total"

NO_OF_PTU = 192

MODEL_FEATURE_SIZE = 44
MODEL_NUM_LAYERS = 4

EMBEDDING_SIZES_TRANSFORMERS = [(31, 15), (12, 6), (24, 12), (4, 2)]
