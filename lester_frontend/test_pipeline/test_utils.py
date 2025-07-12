import pandas as pd
import numpy as np
import re
import torch.nn as nn

def test_dataprep(_lester_dataprep, source_paths):

    tracked_df = _lester_dataprep(**source_paths)
    expected_output = pd.read_csv('./lester_frontend/test_pipeline/expected_output/data_prep_output.csv')

    assert expected_output.shape == tracked_df.df.shape, f"Shape mismatch. EO shape: {expected_output.shape}, TDF shape: {tracked_df.df.shape}"
    
    # assert list(expected_output.columns) == list(tracked_df.df.columns), f"Column mismatch. EO columns: {expected_output.columns}, TDF columns: {tracked_df.df.columns}"

    # Compare DataFrames
    # res = expected_output.compare(tracked_df.df)
    # if not res.empty:
    #     res.to_csv("../test_pipeline/error_logs/tracked_df_differences.csv")
    #     print("Differences saved to df_differences.csv")

    # assert res.empty

    # # Check row_provenance_columns
    # assert isinstance(tracked_df.row_provenance_columns, list)
    # for item in tracked_df.row_provenance_columns:
    #     assert item.startswith("__lester_provenance_")
    #     assert re.match(r"__lester_provenance_0x[0-9a-f]+", item)

    # # Check column_provenance keys and structure
    # expected_keys = {"bank", "country", "is_premium", "title", "text", "sentiment"}
    # assert set(tracked_df.column_provenance.keys()) == expected_keys
    # for key, value_list in tracked_df.column_provenance.items():
    #     assert isinstance(value_list, list)
    #     for v in value_list:
    #         assert re.match(r"0x[0-9a-f]+\.\w+", v)

def test_feature_transformer(encode_features, extract_label):

    print("test_feature_transformer running...")
    train_df = pd.read_csv('./lester_frontend/test_pipeline/inputs/feature_train_input.csv')

    feature_transformer = encode_features()
    feature_transformer = feature_transformer.fit(train_df)

    X_train = pd.DataFrame(feature_transformer.transform(train_df))
    y_train = pd.DataFrame(extract_label(train_df))

    expected_x_train = pd.read_csv('./lester_frontend/test_pipeline/expected_output/x_train.csv', header=None)
    expected_y_train = pd.read_csv('./lester_frontend/test_pipeline/expected_output/y_train.csv', header=None)

    assert X_train.shape == expected_x_train.shape, f"Shape mismatch. X_train shape: {X_train.shape}, Expected shape: {expected_x_train.shape}"
    
    # assert np.allclose(X_train.mean(), expected_x_train.mean(), atol=1e-4), f"Mean of x_train {X_train.mean()} is not the same as expected_x_train {expected_x_train.mean()}"

    # pd.testing.assert_frame_equal(X_train, expected_x_train)
    # pd.testing.assert_frame_equal(y_train, expected_y_train)

    # test_df = pd.read_csv('./lester_frontend/test_pipeline/inputs/feature_test_input.csv')

    # X_test = pd.DataFrame(feature_transformer.transform(test_df))
    # y_test = pd.DataFrame(extract_label(test_df))

    # expected_x_test = pd.read_csv('./lester_frontend/test_pipeline/expected_output/x_test.csv', header=None)
    # expected_y_test = pd.read_csv('./lester_frontend/test_pipeline/expected_output/y_test.csv', header=None)

    # assert X_test.shape == expected_x_test.shape, f"Shape mismatch. X_train shape: {X_test.shape}, Expected shape: {expected_x_test.shape}"
    # assert np.allclose(X_test.mean(), expected_x_test.mean(), atol=1e-4), f"Mean of x_test {X_test.mean()} is not the same as expected expected_x_train {expected_x_test.mean()}"

    # pd.testing.assert_frame_equal(X_test, expected_x_test)
    # pd.testing.assert_frame_equal(y_test, expected_y_test)

def test_model(__model, ):

    expected_x_train = pd.read_csv('./lester_frontend/test_pipeline/expected_output/x_train.csv', header=None)
    num_features = expected_x_train.shape[1]

    model, loss = __model(num_features)

    assert isinstance(model, nn.Module), "Model is not an instance of nn.Module"
    assert isinstance(loss, nn.BCELoss), "Loss is not an instance of nn.BCELoss"
    assert model.__class__.__name__ == "LogisticRegressionModel", "Model defined is not named Logistic Regression Model"