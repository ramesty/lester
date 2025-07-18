from collections import defaultdict
from lester_frontend.test_pipeline.test_utils import (
    run_dataprep_tests, 
    run_featurisation_tests, 
    run_model_tests, 
    handle_error
    )

from lester_frontend.rewrite.seperation_utils import read_synthesized_code

test_functions = {
    "test_data_preperation": run_dataprep_tests,
    "test_featurisation": run_featurisation_tests,
    "test_model": run_model_tests,
}

regenerate_mapping = {
    "regenerate_data_preperation" : "DATAPREP_SYNTHESIZED",
    "regenerate_featurisation": "FEATURE_SYNTHESIZED",
    "regenerate_model": "MODEL_SYNTHESIZED"
}

def rename_keys(highlight_map):

    key_mapping = {
        'green': 'data_preparation',
        'yellow': 'data_featurisation',
        'red': 'model_training'
    }

    for colour in highlight_map:
        highlight_map[colour] = key_mapping[highlight_map[colour]]


def extrapolate_stage_lines(highlight_map):
    pipeline_stage_lines = defaultdict(list)

    for line_str, color in highlight_map.items():
        line_num = int(line_str)
        pipeline_stage_lines[color].append(line_num)

    pipeline_stage_lines = dict(pipeline_stage_lines)
    return pipeline_stage_lines


def join_code_from_buckets(code_split):
    
    code_stages = {}

    for key, lines in code_split.items():
        joined_lines = "\n".join(lines)
        code_stages[key] = joined_lines

    return code_stages

def split_code_by_stage(code_lines, color_line_map):
    color_buckets = {}

    for color, lines in color_line_map.items():
        color_buckets[color] = [code_lines[line_num - 1].rstrip() for line_num in lines]

    code_stages = join_code_from_buckets(color_buckets)

    return code_stages

async def format_response():
    synth_stages = read_synthesized_code()
    response = { 
        "green" : synth_stages["DATAPREP_SYNTHESIZED"],
        "yellow" : synth_stages["FEATURE_SYNTHESIZED"],
        "red" : synth_stages["MODEL_SYNTHESIZED"]
        }
    
    formatted_response = [
        {"colour": colour, "line" : line}
        for colour, code in response.items()
        for line in code.splitlines()
    ]

    print(formatted_response)
    
    return formatted_response
