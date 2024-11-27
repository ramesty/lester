import os
import warnings
from synthesised_code import SYNTHESISED_CREDITCARD_DATAPREP_CODE, SYNTHESISED_CREDITCARD_FEATURISATION_CODE, \
    SYNTHESISED_SKLEARNLOGREG_CODE
from lester.classification import run_pipeline, instantiate

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "False"


# We don't have support for rewriting the label extraction code yet...
def extract_label(df):
    import numpy as np
    label = np.where((df['sentiment'] == 'negative') & (df['is_premium'] == True), 1.0, 0.0)
    return label


source_paths = {
    'customers_file': 'data/synthetic_customers_10.csv',
    'mails_file': 'data/synthetic_mails_10.csv',
}

__dataprep = instantiate("__dataprep", SYNTHESISED_CREDITCARD_DATAPREP_CODE)
__featurise = instantiate("__featurise", SYNTHESISED_CREDITCARD_FEATURISATION_CODE)
__model = instantiate("__model", SYNTHESISED_SKLEARNLOGREG_CODE)

run_pipeline("lester-synth", source_paths, __dataprep, __featurise, extract_label, __model)
