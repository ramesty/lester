import json
import ast

# read/write
def convert_py_to_json(input_path, output_path=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    code_lines = [{"line": i + 1, "code": line.rstrip()} for i, line in enumerate(lines)]

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(code_lines, out_file, indent=2)
    else:
        print(json.dumps(code_lines, indent=2))

# read/write
async def read_synthesized_code():
    with open("./lester_frontend/pipeline_stages/synthesized_stages.json", "r") as f:
        return json.load(f)
    
# read/write
async def read_synthesized_code_stage(stage_name):
    with open("./lester_frontend/pipeline_stages/synthesized_stages.json", "r") as f:
        synth_code = json.load(f)

    return synth_code[stage_name]

# read/write
async def write_synthesized_code_stage(stage_name, new_stage_code):
    with open("./lester_frontend/pipeline_stages/synthesized_stages.json", "r") as f:
        synth_code = json.load(f)

    synth_code[stage_name] = new_stage_code

# read/write
async def read_original_code():
    with open("./lester_frontend/pipeline_stages/original_stages.json", "r") as f:
        return json.load(f) 

# read/write
async def write_original_code_to_file(original_stages):

    print("Writing original code to file...")

    # Write to JSON file
    with open("./lester_frontend/pipeline_stages/original_stages.json", "w") as out_json:
        json.dump(original_stages, out_json, indent=2)

    with open("./lester_frontend/pipeline_stages/logs/original_stages.txt", "w") as out_txt:
        for key, value in original_stages.items():
            out_txt.write(f"\n ----------------------------------------------- \n\n{key}:\n\n")
            out_txt.write(value)

# read/write
async def write_synthesized_code_to_file(synthesized_stages):

    print("Writing synthesized code to file...")
    path = "./lester_frontend/pipeline_stages/synthesized_stages.json"

    with open(path, "w") as out:
        json.dump(synthesized_stages, out, indent=2)

    with open("./lester_frontend/pipeline_stages/logs/synthesized_stages.txt", "w") as out_txt:
        for key, value in synthesized_stages.items():
            out_txt.write(f"\n ----------------------------------------------- \n\n{key}:\n\n")
            out_txt.write(value)

# read/write
async def update_synthesized_json_stage(stage_name, stage_code):
    path = "./lester_frontend/pipeline_stages/synthesized_stages.json"

    with open(path, "r") as f:
        existing_data = json.load(f)

    existing_data[stage_name] = stage_code

    with open(path, "w") as out:
        json.dump(existing_data, out, indent=2)

# read/write
async def read_error_msg_json(error_stage_name):

    with open("./lester_frontend/pipeline_stages/error_msg.json", "r") as f:
        existing_errors = json.load(f)

    return existing_errors[error_stage_name]

# read/write
async def add_dataprep():
    return

# read / write 
def write_payload(code_lines, highlight_map, manual_inputs, pipeline_stage_lines, code_stages):
    
    # Combine into a dictionary
    data_to_save = {
        "code_lines": code_lines,
        "highlight_map": highlight_map,
        "manual_inputs": manual_inputs,
        "pipeline_stage_lines" : pipeline_stage_lines,
        "code_stages" : code_stages
    }
    # Save to a file
    with open("./lester_frontend/backend/saved_payload.json", "w") as f:
        json.dump(data_to_save, f, indent=2)

# read/write
def append_synthesized_iteration_log(stage, iteration_code):
    with open(f"./lester_frontend/pipeline_stages/logs/synthesized_iterations/{stage}.txt", "a") as f:
        f.write(f"\n ----------------------------------------------- \n\n{stage}:\n\n")
        f.write(iteration_code)   

# read/write
def update_error_msg_json(error_stage_name, error_msg):

    path = "./lester_frontend/pipeline_stages/error_msg.json"

    with open(path, "r") as f:
        existing_errors = json.load(f)

    existing_errors[error_stage_name] = error_msg

    with open(path, "w") as out_json:
        json.dump(existing_errors, out_json, indent=2)  