import gradio as gr
from lester_frontend.frontend.gradio_utils import (
    update_dropdown,
    load_sample_inputs,
    run_pipeline_with_model,
    on_start,
    options_map
)

def build_sidebar():
    with gr.Sidebar():
        gr.Markdown("### Lester")

        benchmark_group = gr.Dropdown(
            label="Select Benchmark Group",
            choices=list(options_map.keys())
        )

        task_selector = gr.Dropdown(
            label="Select Example Task",
            choices=list(options_map["Data Preperation"])
        )

        user_input_schema = gr.Textbox(label="Input Schema (CSV format)", lines=3)
        user_input_args = gr.Textbox(label="Input Arguments")
        user_input_out_cols = gr.Textbox(label="Expected Output Columns")
        run_button = gr.Button("Run")

    return benchmark_group, task_selector, user_input_schema, user_input_args, user_input_out_cols, run_button


def build_code_columns():
    with gr.Row(equal_height=True):
        with gr.Column(min_width=400):
            code_input = gr.Code(label="Function", language="python", lines=48)
        with gr.Column(min_width=400):
            code_output = gr.Code(label="Output", language="python", lines=48)
    return code_input, code_output


with gr.Blocks(title="Lester") as demo:
    # Sidebar
    (
        benchmark_group,
        task_selector,
        user_input_schema,
        user_input_args,
        user_input_out_cols,
        run_button
    ) = build_sidebar()

    # Main content
    code_input, code_output = build_code_columns()

    # Interactions
    benchmark_group.change(
        fn=update_dropdown,
        inputs=benchmark_group,
        outputs=[task_selector, user_input_schema, user_input_args, user_input_out_cols, run_button]
    )

    task_selector.change(
        fn=load_sample_inputs,
        inputs=task_selector,
        outputs=[code_input, user_input_schema, user_input_args, user_input_out_cols]
    )

    run_button.click(
        fn=run_pipeline_with_model,
        inputs=[task_selector, code_input, user_input_schema, user_input_args, user_input_out_cols],
        outputs=[code_output]
    )

    demo.load(
        fn=on_start,
        inputs=None,
        outputs=[code_input, user_input_schema, user_input_args, user_input_out_cols]
    )

demo.launch()
