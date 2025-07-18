import json
import traceback

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from lester_frontend.test_pipeline.test_utils import test_dataprep, test_feature_transformer, test_model
from lester.classification import run_pipeline, instantiate


from lester_frontend.rewrite.seperation_utils import (
    convert_py_to_json,
    automate_split_pipeline_stages,
    generate_synthesized_pipeline,
    generate_synthesized_dataprep,
    generate_synthesized_feature,
    generate_synthesized_model,
    handle_error,
    read_original_code,
    read_synthesized_code,
    assign_code_to_stage,
    extract_label,
    source_paths,
)


if __name__ == "__main__":

    load_dotenv()

    # Only use if you want to seperate the original messy pipeline into lines once more
    # convert_py_to_json("../messy_original_pipeline.py", "./json_output/output.json")

    model = init_chat_model("gpt-4o", model_provider="openai")

    # stages = automate_split_pipeline_stages(model)

    # with open("./lester_frontend/pipeline_stages/stage_to_lines.json", "r") as f:
    #     stages = json.load(f)

    # original_stages = assign_code_to_stage(stages)

    original_stages = read_original_code()
    # synthesized_stages = read_synthesized_code()

    if False:
        synth_dp = generate_synthesized_dataprep(original_stages["DATAPREP_ORIGINAL"], model)
        __dataprep = instantiate("__dataprep", synth_dp)

        attempts = 0
        dataprep_works = False

        while attempts < 3:
            if not dataprep_works:
                print("Testing if the dataprep phase works")
                try:
                    test_dataprep(__dataprep, source_paths)
                    print("Dataprep Phase works!")
                    dataprep_works = True

                except Exception as error:
                    print(f"Dataprep has failed: {error}")
                    synth_dp = handle_error(model, synth_dp, error, "DATAPREP_SYNTHESIZED")
                    __dataprep = instantiate("__dataprep", synth_dp)
            attempts += 1

    if True:
        synth_feat = generate_synthesized_feature(original_stages["FEATURE_ORIGINAL"], model)
        # synth_feat = synthesized_stages["FEATURE_SYNTHESIZED"]
        __featurise = instantiate("__featurise", synth_feat)

        attempts = 0
        featurisation_works = False

        while attempts < 3:

            if not featurisation_works:
                print("Testing if the featurisation phase works")
                try:
                    test_feature_transformer(__featurise, extract_label)
                    print("Featurisation Phase works!")
                    featurisation_works = True

                except Exception as error:
                    tb = traceback.format_exc()
                    print("Traceback:")
                    print(tb)
                    synth_feat = handle_error(model, synth_feat, error, "FEATURE_SYNTHESIZED")
                    __featurise = instantiate("__featurise", synth_feat)
            attempts += 1

    # synth_model = generate_synthesized_model(stage_to_code_dict["MODEL_SYNTHESIZED"])

    if False:
        stage_code_dict = assign_code_to_stage(stages)
        generate_synthesized_pipeline(stage_code_dict["data_preparation"], stage_code_dict["data_featurisation"], stage_code_dict["model_training"], model)

    if False:

        all_phases = read_synthesized_code()

        __dataprep = instantiate("__dataprep", all_phases["DATAPREP_SYNTHESIZED"])
        __featurise = instantiate("__featurise", all_phases["FEATURE_SYNTHESIZED"])
        __model = instantiate("__model", all_phases["MODEL_SYNTHESIZED"])

        dataprep_works = False
        featurisation_works = False
        model_works = False

        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:

            if not dataprep_works:
                print("Testing if the dataprep phase works")
                try:
                    test_dataprep(__dataprep, source_paths)
                    print("Dataprep Phase works!")
                    dataprep_works = True

                except Exception as error:
                    print(f"Dataprep has failed: {error}")
                    handle_error(model, all_phases["DATAPREP_SYNTHESIZED"], error, "DATAPREP_SYNTHESIZED")
                    __dataprep = instantiate("__dataprep", all_phases["DATAPREP_SYNTHESIZED"])

            if not featurisation_works:
                print("Testing if the featurisation phase works")
                try:
                    test_feature_transformer(__featurise, extract_label)
                    print("Featurisation Phase works!")
                    featurisation_works = True

                except Exception as error:
                    print(f"Featurisation has failed: {error}")
                    handle_error(model, all_phases["FEATURE_SYNTHESIZED"], error, "FEATURE_SYNTHESIZED")
                    __featurise = instantiate("__featurise", all_phases["FEATURE_SYNTHESIZED"])

            if not model_works:
                print("Testing if the model training phase works")
                try:
                    test_model(__model)
                    print("Model Training Phase works!")
                    model_works = True

                except Exception as error:
                    print(f"Model training has failed: {error}")
                    handle_error(model, all_phases["MODEL_SYNTHESIZED"], error, "MODEL_SYNTHESIZED")
                    __model = instantiate("__model", all_phases["MODEL_SYNTHESIZED"])

            if featurisation_works: # and featurisation_works and model_works:
                break

            attempts += 1
            print(f'Attempts: {attempts}')




