from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from lester_frontend.rewrite.seperation_utils import test_generate_synthesized_pipeline, generate_synthesized_pipeline, read_synthesized_code
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from lester_frontend.backend.backend_utils import test_functions, regenerate_mapping, rename_keys, extrapolate_stage_lines, split_code_by_stage, format_response

from lester_frontend.test_pipeline.test_utils import (
    run_dataprep_tests, 
    run_featurisation_tests, 
    run_model_tests, 
    handle_error
    )

@asynccontextmanager
async def lifeSpan(app: FastAPI):
    load_dotenv()
    model = init_chat_model("gpt-4o", model_provider="openai")
    app.state.model = model
    yield


app = FastAPI(lifespan=lifeSpan)

# Allow requests from your frontend's origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/test_stage/{stage_name}")
async def test_code(stage_name: str):
    test_fnc = test_functions.get(stage_name)
    if test_fnc:
        return await test_fnc()
    return {"error" : "Unknown stage selected."}                   

# idea, regenerate the problematic code
# read all the code again, and send the entire pipeline back

@app.post("/regenerate_stage/{stage_name}")
async def regenerate_stage(stage_name: str):
    reg_stage = regenerate_mapping.get(stage_name)
    if reg_stage:
        return await format_response()
    return {"error" : "Unknown stage selected"}
    
@app.post("/run")
async def run_code(payload: dict):

    model = app.state.model

    code_lines = payload['code'].splitlines()
    highlight_map = payload['highlightMap']
    manual_inputs = payload['manualInputs']

    rename_keys(highlight_map)
    pipeline_stage_lines = extrapolate_stage_lines(highlight_map)
    code_stages = split_code_by_stage(code_lines, pipeline_stage_lines)

    # Combine into a dictionary
    # data_to_save = {
    #     "code_lines": code_lines,
    #     "highlight_map": highlight_map,
    #     "manual_inputs": manual_inputs,
    #     "pipeline_stage_lines" : pipeline_stage_lines,
    #     "code_stages" : code_stages
    # }

    # Save to a file
    # with open("./backend/saved_payload.json", "w") as f:
    #     json.dump(data_to_save, f, indent=2)

    # d_obj, f_obj, m_obj = test_generate_synthesized_pipeline(code_stages, manual_inputs, model)

    generate_synthesized_pipeline(code_stages['data_preparation'], code_stages['data_featurisation'], code_stages['model_training'], model)

    return format_response()
