from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict

from lester_frontend.rewrite.seperation_utils import test_generate_synthesized_pipeline
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import json


app = FastAPI()

# Allow requests from your frontend's origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
@app.post("/run")
async def run_code(payload: dict):
    # your logic here

    code_lines = payload['code'].splitlines()
    highlight_map = payload['highlightMap']
    manual_inputs = payload['manualInputs']

    rename_keys(highlight_map)
    pipeline_stage_lines = extrapolate_stage_lines(highlight_map)
    code_stages = split_code_by_stage(code_lines, pipeline_stage_lines)

    # Combine into a dictionary
    data_to_save = {
        "code_lines": code_lines,
        "highlight_map": highlight_map,
        "manual_inputs": manual_inputs,
        "pipeline_stage_lines" : pipeline_stage_lines,
        "code_stages" : code_stages
    }

    # Save to a file
    # with open("./backend/saved_payload.json", "w") as f:
    #     json.dump(data_to_save, f, indent=2)

    load_dotenv()

    model = init_chat_model("gpt-4o", model_provider="openai")

    d_obj, f_obj, m_obj = test_generate_synthesized_pipeline(code_stages, manual_inputs, model)

    response = {
        d_obj.colour : d_obj.synthesized_code,
        f_obj.colour : f_obj.synthesized_code,
        m_obj.colour : m_obj.synthesized_code 
    }

    formatted_response = [
        {"colour": colour, "line": line}
        for colour, code in response.items()
        for line in code.splitlines()
    ]

    print(formatted_response)


    return formatted_response
