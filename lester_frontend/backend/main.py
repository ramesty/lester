from fastapi import FastAPI
import json
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from lester_frontend.backend.utils.io_utils import write_payload, read_synthesized_code
from lester_frontend.backend.utils.format_utils import rename_keys, extrapolate_stage_lines, split_code_by_stage, format_response
from lester_frontend.backend.utils.generate_utils import handle_regenerate_stage, generate_synthesized_pipeline, handle_error

from lester_frontend.backend.utils.test_utils import (
    run_dataprep_tests, 
    run_featurisation_tests, 
    run_model_tests, 
    )

# test
test_functions = {
    "test_data_preperation": run_dataprep_tests,
    "test_featurisation": run_featurisation_tests,
    "test_model": run_model_tests,
}

# generate
regenerate_mapping = {
    "regenerate_data_preperation" : "DATAPREP_SYNTHESIZED",
    "regenerate_featurisation": "FEATURE_SYNTHESIZED",
    "regenerate_model": "MODEL_SYNTHESIZED"
}

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

@app.post("/regenerate_stage/{stage_name}")
async def regenerate_stage(stage_name: str):
    reg_stage = regenerate_mapping.get(stage_name)
    if reg_stage:
        return await handle_regenerate_stage(app.state.model, reg_stage)
    return {"error" : "Unknown stage selected"}

@app.post("/run")
async def run_code(payload: dict):

    model = app.state.model

    code_lines = payload['inputCode'].splitlines()
    highlight_map = payload['highlightMap']
    manual_inputs = payload['manualInputs']

    rename_keys(highlight_map)
    pipeline_stage_lines = extrapolate_stage_lines(highlight_map)
    code_stages = split_code_by_stage(code_lines, pipeline_stage_lines)

    # d_obj, f_obj, m_obj = test_generate_synthesized_pipeline(code_stages, manual_inputs, model)
    # write_payload(code_lines, highlight_map, manual_inputs, pipeline_stage_lines, code_stages)

    await generate_synthesized_pipeline(code_stages['data_preparation'], code_stages['data_featurisation'], code_stages['model_training'], model)
    return await format_response()
