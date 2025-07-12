# Standard library imports
import os

# Third-party library imports
import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Application-specific imports
from lester_frontend.frontend.LLM_task_classes import (
    LLMDataprepTask,
    LLMFeaturisationTask,
    LLMModelCodeTransformationTask
)

from lester.rewrite import (
    generate_dataprep_code,
    generate_featurisation_code,
    generate_model_code
)

# Sample input modules (for testing)
from sample_inputs.data_prep_sample_code import (
    dataprep_original_code,
    dataprep_input_arg_names,
    dataprep_input_schemas,
    dataprep_output_columns
)
from sample_inputs.featurisation_sample_code import featurisation_original_code, featurisation_input_schema
from sample_inputs.model_code_sample_code import model_code_sample_code

options_map = {
    "Data Preperation": ["CreditcardDataprepTask"],
    "Featurisation": ["CreditcardFeaturisationTask"],
    "Model Development": ["SKlearnlogregModelTask"],
}

def on_start():
    return load_sample_inputs("CreditcardDataprepTask")

def create_model():
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables.")
    return init_chat_model("gpt-4o", model_provider="openai")

def run_pipeline_with_model(task_choice, code_input, schema, args, output_cols):
    model = create_model()
    return run_pipeline(task_choice, model, code_input, schema, args, output_cols)

def update_dropdown(benchmark_group):
    print(benchmark_group)
    
    # Update task_selector dropdown choices
    tasks = options_map.get(benchmark_group, [])
    task_selector_update = gr.update(choices=tasks, value=tasks[0] if tasks else None)
    
    # Logic for toggling textbox visibility
    show_schema = show_args = show_outputs = run_visible = False
    if benchmark_group == "Data Preperation":
        show_schema = show_args = show_outputs = run_visible = True
    elif benchmark_group == "Featurisation":
        show_schema = run_visible = True
    elif benchmark_group == "Model Development":
        run_visible = True

    return (
        task_selector_update,                      # 1. Update task selector dropdown
        gr.update(visible=show_schema),            # 2. Input schema
        gr.update(visible=show_args),              # 3. Input args
        gr.update(visible=show_outputs),           # 4. Output cols
        gr.update(visible=run_visible)             # 5. Run button
    )

def load_sample_inputs(task_choice):
    print(f"Loading inputs for {task_choice}")
    
    match task_choice:
        case "CreditcardDataprepTask":
            return (
                dataprep_original_code,
                dataprep_input_arg_names,
                dataprep_input_schemas,
                dataprep_output_columns
                )
        case "CreditcardFeaturisationTask":
            return (
                featurisation_original_code,
                featurisation_input_schema,
                "",
                ""
                )
        case "SKlearnlogregModelTask":
            return (
                model_code_sample_code,
                "",
                "",
                ""
                )
        case _:
            return ("", "", "", "")

def run_pipeline(task_choice, model, code_input, user_input_schemas, user_input_args, user_input_out_cols):
    
    print(task_choice + " running.")

    match task_choice:
        case "CreditcardDataprepTask":
            data_task = LLMDataprepTask(
                original_code=code_input,
                input_schemas = user_input_schemas,
                input_arg_names=user_input_args,
                output_columns=user_input_out_cols
            )
            dataprep_synthesized_code = generate_dataprep_code(data_task, model)

            return dataprep_synthesized_code
        
        case "CreditcardFeaturisationTask":
            feature_task = LLMFeaturisationTask(
                original_code=code_input,
                input_schemas=user_input_schemas
            )

            feature_synthesized_code = generate_featurisation_code(feature_task, model)

            return feature_synthesized_code
        case "SKlearnlogregModelTask":
            model_task = LLMModelCodeTransformationTask(
                original_code=code_input
            )

            model_synthesized_code = generate_model_code(model_task, model)

            return model_synthesized_code
        
        case _:
            return "No task selected.", ""