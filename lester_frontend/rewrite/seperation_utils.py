import ast
import json
import os

from lester.rewrite import generate_dataprep_code, generate_featurisation_code, generate_model_code, regenerate_code
from lester_frontend.sample_inputs.sample_code import messy_original_pipeline, dataprep_input_arg_names, dataprep_input_schemas, dataprep_output_columns, featurisation_input_schema
from lester_frontend.LLM_task_classes import LLMDataprepTask, LLMFeaturisationTask, LLMModelCodeTransformationTask

source_paths = {
    'customers_file': './data/synthetic_customers_10.csv',
    'mails_file': './data/synthetic_mails_10.csv',
    }

def extract_label(df):
    import numpy as np
    label = np.where((df['sentiment'] == 'negative') & (df['is_premium'] == True), 1.0, 0.0)
    return label

def convert_py_to_json(input_path, output_path=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    code_lines = [{"line": i + 1, "code": line.rstrip()} for i, line in enumerate(lines)]

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(code_lines, out_file, indent=2)
    else:
        print(json.dumps(code_lines, indent=2))

def extract_code(response):
    generated_code = response.content
    if '```json' in generated_code:
        generated_code = generated_code.split('```json')[1].split('```')[0]
    try:
        ast.parse(generated_code)
        return generated_code
    except Exception as e:
        print(f"SYNTACTICALLY INCORRECT CODE GENERATED:\n\n{e}\\n\n{generated_code}")

def read_synthesized_code():
    with open("./lester_frontend/pipeline_stages/synthesized_stages.json", "r") as f:
        return json.load(f)

def read_original_code():
    with open("./lester_frontend/pipeline_stages/original_stages.json", "r") as f:
        return json.load(f) 

def write_original_code_to_file(original_stages):

    print("Writing original code to file...")

    # Write to JSON file
    with open("./lester_frontend/pipeline_stages/original_stages.json", "w") as out_json:
        json.dump(original_stages, out_json, indent=2)

    with open("./lester_frontend/pipeline_stages/logs/original_stages.txt", "w") as out_txt:
        for key, value in original_stages.items():
            out_txt.write(f"\n ----------------------------------------------- \n\n{key}:\n\n")
            out_txt.write(value)

def write_synthesized_code_to_file(synthesized_stages):

    print("Writing synthesized code to file...")
    path = "./lester_frontend/pipeline_stages/synthesized_stages.json"

    with open(path, "w") as out:
        json.dump(synthesized_stages, out, indent=2)

    with open("./lester_frontend/pipeline_stages/logs/synthesized_stages.txt", "w") as out_txt:
        for key, value in synthesized_stages.items():
            out_txt.write(f"\n ----------------------------------------------- \n\n{key}:\n\n")
            out_txt.write(value)

def append_iteration(stage, iteration_code):
    with open(f"./lester_frontend/pipeline_stages/logs/synthesized_iterations/{stage}.txt", "a") as f:
        f.write(f"\n ----------------------------------------------- \n\n{stage}:\n\n")
        f.write(iteration_code)
        

def update_synthesized_json(synthesized_stages):
    path = "./lester_frontend/pipeline_stages/synthesized_stages.json"

    with open(path, "r") as f:
        existing_data = json.load(f)

    existing_data.update(synthesized_stages)

    with open(path, "w") as out:
        json.dump(existing_data, out, indent=2)


def add_dataprep():
    return

def create_dictionary(dataprep_code, featurisation_code, model_code, type):
    
    stages = {
        f"DATAPREP_{type}": dataprep_code,
        f"FEATURE_{type}": featurisation_code,
        f"MODEL_{type}": model_code
    }
    return stages

def handle_error(model, ERROR_SYNTHESISED_CODE, error,  SYNTHESIZED_TITLE):

    print("Handling error...")
    synthesized_stages = read_synthesized_code()
    synthesized_stages[SYNTHESIZED_TITLE] = regenerate_code(model, ERROR_SYNTHESISED_CODE, error)
    write_synthesized_code_to_file(synthesized_stages)
    return synthesized_stages[SYNTHESIZED_TITLE]

def generate_synthesized_dataprep(dataprep_org_code, model):

    data_task = LLMDataprepTask(dataprep_org_code, dataprep_input_arg_names, dataprep_input_schemas, dataprep_output_columns)
    dataprep_code = generate_dataprep_code(data_task, model)
    append_iteration("dataprep", dataprep_code)
    return dataprep_code

def generate_synthesized_feature(feature_org_code, model):

    feature_task = LLMFeaturisationTask(feature_org_code, featurisation_input_schema)
    featurisation_code = generate_featurisation_code(feature_task, model)
    append_iteration("feature", featurisation_code)
    return featurisation_code

def generate_synthesized_model(model_org_code, model):
    model_task = LLMModelCodeTransformationTask(model_org_code)
    model_code = generate_model_code(model_task, model)
    return model_code

def generate_synthesized_pipeline(dataprep_org_code, feature_org_code, model_org_code, model):
    
    dataprep_code = generate_synthesized_dataprep(dataprep_org_code, model)
    featurisation_code = generate_synthesized_feature(feature_org_code, model)
    model_code = generate_synthesized_model(model_org_code, model)
    synthesized_stages = create_dictionary(dataprep_code, featurisation_code, model_code, "SYNTHESIZED")
    write_synthesized_code_to_file(synthesized_stages)

def test_generate_synthesized_pipeline(code_stages, inputs, model):

    dataprep_org_code = code_stages.get('data_preparation')
    feature_org_code = code_stages.get('data_featurisation')
    model_org_code = code_stages.get('model_training')


    dataprep_input_arg_names = inputs[0]
    dataprep_input_schemas = inputs[1]
    dataprep_output_columns = inputs[2]
    featurisation_input_schema = inputs[3]

    data_task = LLMDataprepTask(dataprep_org_code, dataprep_input_arg_names, dataprep_input_schemas, dataprep_output_columns)
    feature_task = LLMFeaturisationTask(feature_org_code, featurisation_input_schema)
    model_task = LLMModelCodeTransformationTask(model_org_code)

    # Assume these functions return the code snippets as strings
    dataprep_code = generate_dataprep_code(data_task, model)
    featurisation_code = generate_featurisation_code(feature_task, model)
    model_code = generate_model_code(model_task, model)

    data_task.set_synthesized_code(dataprep_code)
    feature_task.set_synthesized_code(featurisation_code)
    model_task.set_synthesized_code(model_code)

    # For testing Purposes
    # data_task.set_synthesized_code("test data synthesized code\nTest second line of code\nthird line of code")
    # feature_task.set_synthesized_code("test feature synthesized code\n secodn line")
    # model_task.set_synthesized_code("test model synthesized code")

    return data_task, feature_task, model_task

def automate_split_pipeline_stages(model):

    # Load your prompt template
    with open("./lester_frontend/rewrite/my_prompt.txt", "r") as f:
        prompt_template = f.read()

    # Load your line seperated code
    with open("./lester_frontend/json_output/output.json", "r") as f:
        code_json = json.load(f)

    code_json_str = json.dumps(code_json, indent=2)
    final_prompt = prompt_template.replace("{code_json}", code_json_str)
    response = model.invoke(final_prompt)
    json_response = extract_code(response)
    stages = json.loads(json_response)
    
    with open ("lester_frontend/pipeline_stages/stage_to_lines.json", "w") as stages_f:
        json.dump(stages, stages_f, indent=2)

    # print(f"ML Pipeline has been seperated into the following phases: {stages}")

    return stages

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
    