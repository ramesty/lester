import os
import sys

sys.path.append('../')

import warnings
import json
from test_pipeline.classification import run_pipeline, instantiate
from test_pipeline.test_utils import test_dataprep, test_feature_transformer, test_model

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "False"


# We don't have support for rewriting the label extraction code yet...
def extract_label(df):
    import numpy as np
    label = np.where((df['sentiment'] == 'negative') & (df['is_premium'] == True), 1.0, 0.0)
    return label


source_paths = {
    'customers_file': '../data/synthetic_customers_10.csv',
    'mails_file': '../data/synthetic_mails_10.csv',
}

# Open and read the JSON file
with open('./phase_output/all_phases.json', 'r') as file:
    all_phases = json.load(file)

dataprep_synthesized = all_phases.get('DATAPREP_SYNTHESIZED')
feature_synthesized = all_phases.get('FEATURE_SYNTHESIZED')
model_synthesized = all_phases.get('MODEL_SYNTHESIZED')

__dataprep = instantiate("__dataprep", dataprep_synthesized)
__featurise = instantiate("__featurise", feature_synthesized)
__model = instantiate("__model", model_synthesized)

test_dataprep(__dataprep, source_paths)
test_feature_transformer(__featurise, extract_label)
test_model(__model)

run_pipeline("lester-synth", source_paths, __dataprep, __featurise, extract_label, __model)
