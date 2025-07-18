import ast
from langchain_core.prompts import PromptTemplate
from lester.rewrite.prompts import DATAPREP_COT, FIX, FEATURISATION_COT, MODEL_COT, DATAPREP_COT_TEST, FEATURISATION_COT_TEST


def extract_code(response):
    generated_code = response.content
    if '```python' in generated_code:
        generated_code = generated_code.split('```python')[1].split('```')[0]
    try:
        ast.parse(generated_code)
        return generated_code
    except Exception as e:
        print(f"SYNTACTICALLY INCORRECT CODE GENERATED:\n\n{e}\\n\n{generated_code}")


def generate_dataprep_code(task, model):

    with open("lester/__init__.py", "r") as lib_file:
        ld_lib_code = lib_file.read()
        lib_file.close()

    with open("messy_original_pipeline.py", "r") as file:
        messy_code = file.read()
        file.close()

    if len(task.input_schemas()) == 1:
        hint = f"The schema of the input data for the code is: {','.join(task.input_schemas()[0])}."
    else:
        hint = f"The code consumes {len(task.input_schemas())} types of input data with the following schemas: "
        for schema in task.input_schemas():
            hint += f"{','.join(schema)}. "

    params = {
        'lester_lib_code': ld_lib_code,
        'input_args': ', '.join(task.input_arg_names()),
        'input_hint': hint,
        'columns': task.output_columns(),
        'entire_pipeline': messy_code,
        'code': task.original_code
    }

    prompt_template = PromptTemplate.from_template(DATAPREP_COT_TEST)
    prompt = prompt_template.invoke(params)
    response = model.invoke(prompt)
    generated_code = extract_code(response)

    return generated_code


def generate_featurisation_code(task, model):

    with open("messy_original_pipeline.py", "r") as file:
        messy_code = file.read()
        file.close()

    params = {
        'columns': ', '.join(task.input_schema),
        'entire_pipeline' : messy_code,
        'code': task.original_code
    }

    prompt_template = PromptTemplate.from_template(FEATURISATION_COT_TEST)
    prompt = prompt_template.invoke(params)
    response = model.invoke(prompt)
    generated_code = extract_code(response)

    return generated_code


def generate_model_code(task, model):
    params = {
        'code': task.original_code
    }

    prompt_template = PromptTemplate.from_template(MODEL_COT)
    prompt = prompt_template.invoke(params)
    response = model.invoke(prompt)
    generated_code = extract_code(response)

    return generated_code

def write_prompt_to_file(prompt):

    print("Writing prompt code to file...")

    with open("./lester_frontend/rewrite/dp_prompt.txt", "a") as out:
        out.write(("-----------------------------------------"))
        out.write(prompt)

def regenerate_code(model, generated_code, previous_error):

    params = {
        'generated_code': generated_code,
        'error_message': previous_error,
    }
    prompt_template = PromptTemplate.from_template(FIX)
    prompt = prompt_template.invoke(params)
    write_prompt_to_file(prompt.to_string())
    response = model.invoke(prompt)
    generated_code = extract_code(response)
    return generated_code


def try_to_run(model, task, generated_code, previous_error=None, attempts=0):

    if attempts > 1:
        print("Giving up... Here is how far we got:\n")
        print(generated_code)

    else:
        if previous_error is not None:
            params = {
                'generated_code': generated_code,
                'error_message': previous_error,
            }
            prompt_template = PromptTemplate.from_template(FIX)
            prompt = prompt_template.invoke(params)
            response = model.invoke(prompt)
            generated_code = extract_code(response)

        try:
            task.evaluate_transformed_code(generated_code)
            return "ALL GOOD!!!"
        except Exception as e:
            error = repr(e)
            print(f"ERROR: {e}, trying fix")
            print(generated_code)
            print("\n\n")
            try_to_run(model, task, generated_code, error, attempts+1)
