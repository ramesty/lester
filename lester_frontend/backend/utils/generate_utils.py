import json
from lester.rewrite import generate_dataprep_code, generate_featurisation_code, generate_model_code, regenerate_code
from lester_frontend.sample_inputs.sample_code import messy_original_pipeline, dataprep_input_arg_names, dataprep_input_schemas, dataprep_output_columns, featurisation_input_schema
from lester_frontend.LLM_task_classes import LLMDataprepTask, LLMFeaturisationTask, LLMModelCodeTransformationTask

from lester_frontend.backend.utils.io_utils import read_synthesized_code_stage, write_synthesized_code_to_file, append_synthesized_iteration_log, read_error_msg_json, update_synthesized_json_stage
from lester_frontend.backend.utils.format_utils import create_dictionary, extract_code, format_response

# generate
async def handle_error(model, ERROR_SYNTHESISED_CODE, error,  SYNTHESIZED_TITLE):

    print("Handling error...")
    updated_stage = await regenerate_code(model, ERROR_SYNTHESISED_CODE, error)
    await update_synthesized_json_stage(SYNTHESIZED_TITLE, updated_stage)
    return updated_stage

# generate
async def handle_regenerate_stage(model, reg_stage):
    error_code = await read_synthesized_code_stage(reg_stage)
    error_msg = await read_error_msg_json(reg_stage)
    await handle_error(model, error_code, error_msg, reg_stage)
    return await format_response()

# generate
async def generate_synthesized_dataprep(dataprep_org_code, model):

    data_task = LLMDataprepTask(dataprep_org_code, dataprep_input_arg_names, dataprep_input_schemas, dataprep_output_columns)
    dataprep_code = await generate_dataprep_code(data_task, model)
    append_synthesized_iteration_log("dataprep", dataprep_code)
    return dataprep_code

# generate
async def generate_synthesized_feature(feature_org_code, model):

    feature_task = LLMFeaturisationTask(feature_org_code, featurisation_input_schema)
    featurisation_code = await generate_featurisation_code(feature_task, model)
    append_synthesized_iteration_log("feature", featurisation_code)
    return featurisation_code

# generate
async def generate_synthesized_model(model_org_code, model):
    model_task = LLMModelCodeTransformationTask(model_org_code)
    model_code = await generate_model_code(model_task, model)
    return model_code

# generate
async def generate_synthesized_pipeline(dataprep_org_code, feature_org_code, model_org_code, model):
    
    dataprep_code = await generate_synthesized_dataprep(dataprep_org_code, model)
    featurisation_code = await generate_synthesized_feature(feature_org_code, model)
    model_code = await generate_synthesized_model(model_org_code, model)
    synthesized_stages = create_dictionary(dataprep_code, featurisation_code, model_code, "SYNTHESIZED")
    await write_synthesized_code_to_file(synthesized_stages)

# generate
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

# generate
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
