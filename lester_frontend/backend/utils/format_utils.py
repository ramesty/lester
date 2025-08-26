import ast
from collections import defaultdict
from lester_frontend.backend.utils.io_utils import read_synthesized_code, write_original_code_to_file

# format
def rename_keys(highlight_map):

    key_mapping = {
        'green': 'data_preparation',
        'yellow': 'data_featurisation',
        'red': 'model_training'
    }

    for colour in highlight_map:
        highlight_map[colour] = key_mapping[highlight_map[colour]]

# format
def extract_code(response):
    generated_code = response.content
    if '```json' in generated_code:
        generated_code = generated_code.split('```json')[1].split('```')[0]
    try:
        ast.parse(generated_code)
        return generated_code
    except Exception as e:
        print(f"SYNTACTICALLY INCORRECT CODE GENERATED:\n\n{e}\\n\n{generated_code}")

# format
def extrapolate_stage_lines(highlight_map):
    pipeline_stage_lines = defaultdict(list)

    for line_str, color in highlight_map.items():
        line_num = int(line_str)
        pipeline_stage_lines[color].append(line_num)

    pipeline_stage_lines = dict(pipeline_stage_lines)
    return pipeline_stage_lines

# format
def join_code_from_buckets(code_split):
    
    code_stages = {}

    for key, lines in code_split.items():
        joined_lines = "\n".join(lines)
        code_stages[key] = joined_lines

    return code_stages

# format
def split_code_by_stage(code_lines, color_line_map):
    color_buckets = {}

    for color, lines in color_line_map.items():
        color_buckets[color] = [code_lines[line_num - 1].rstrip() for line_num in lines]

    code_stages = join_code_from_buckets(color_buckets)

    return code_stages

# format
def assign_code_to_stage(stages):
    # Read the original Python file
    with open("./messy_original_pipeline.py") as f:
        lines = f.readlines()

    # Predefine the variables as empty strings inside a dict
    stage_to_code_dict = {
        "data_preparation": "",
        "data_featurisation": "",
        "label_extraction": "",
        "model_training": ""
    }

    # Fill the dictionary entries with the corresponding code
    for stage_name, stage_code in stage_to_code_dict.items():
        if stage_name in stages:
            start, end = stages[stage_name]
            stage_to_code_dict[stage_name] = "".join(lines[start - 1:end])
        else:
            stage_to_code_dict[stage_name] = ""

    original_stages = create_dictionary(stage_to_code_dict["data_preparation"] , stage_to_code_dict["data_featurisation"], stage_to_code_dict["model_training"], "ORIGINAL")
    write_original_code_to_file(original_stages)

    return original_stages

# format
def create_dictionary(dataprep_code, featurisation_code, model_code, type):
    
    stages = {
        f"DATAPREP_{type}": dataprep_code,
        f"FEATURE_{type}": featurisation_code,
        f"MODEL_{type}": model_code
    }
    return stages

# format
def extract_code(response):
    generated_code = response.content
    if '```json' in generated_code:
        generated_code = generated_code.split('```json')[1].split('```')[0]
    try:
        ast.parse(generated_code)
        return generated_code
    except Exception as e:
        print(f"SYNTACTICALLY INCORRECT CODE GENERATED:\n\n{e}\\n\n{generated_code}")

# format
async def format_response():

    print("Formatting Response")

    synth_stages = await read_synthesized_code()
    response = { 
        "green" : synth_stages["DATAPREP_SYNTHESIZED"],
        "yellow" : synth_stages["FEATURE_SYNTHESIZED"],
        "red" : synth_stages["MODEL_SYNTHESIZED"]
        }
    
    formatted_response = []

    for colour, code in response.items():
        for line in code.splitlines():
            formatted_response.append({"colour": colour, "line" : line})
    
    return formatted_response