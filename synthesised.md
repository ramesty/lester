# Potential of Code Synthesis

At the core of our proposal is the idea to [rewrite messy code](lester/rewrite/__init__.py) for [various pipeline stages](lester/benchmark/__init__.py) based on LLMs with [custom designed prompts](lester/rewrite/prompts.py). 

We provide example rewrites for [nine different rewriting tasks](lester/benchmark) with their corresponding [synthesised pipeline code](synthesised_code.py). These examples have been rewritten with the help of OpenAI's [gpt-4o](https://openai.com/index/hello-gpt-4o/) model and the [generate_dataprep_code](lester/rewrite/__init__.py#L17), [generate_featurisation_code](lester/rewrite/__init__.py#L40) and [generate_model_code](lester/rewrite/__init__.py#L54) methods in our prototype. The corresponding [prompts](lester/rewrite/prompts.py) are available as well.

## Relational data preparation

| Benchmark task | Source |  Original code | Synthesised code | Correct? | Notes |
|---|---|---|---|---|---|
| [creditcard_dataprep.py](lester/benchmark/creditcard_dataprep.py) | running example in paper | [view original code](lester/benchmark/creditcard_dataprep.py#L7) | [view synthesised code](synthesised_code.py#L2)  |:white_check_mark: ||
| [yichun_dataprep.py](lester/benchmark/yichun_dataprep.py) | [GitHub](https://github.com/YichunAstrid/e-commerce-use-case/tree/main/1116LogisticRegression) | [view original code](lester/benchmark/yichun_dataprep.py#L8) |[view synthesised code](synthesised_code.py#L51) |:x:| Manual editing of two lines required to handle partitioned inputs (the required manual fix is included as comment) |
| [amazonreviews_dataprep.py](lester/benchmark/amazonreviews_dataprep.py) | [GitHub](https://github.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset/blob/master/SVM_model.ipynb) | [view original code](lester/benchmark/amazonreviews_dataprep.py#L8)| [view synthesised code](synthesised_code.py#L81) |:white_check_mark:|Dead code generated,<br/> no impact on final output|

## Feature encoding

| Task with messy original code  | Source |  Code synthesised by LLM | Correct?| Notes |
|---|---|---|---|---|
| [creditcard_featurisation.py](lester/benchmark/creditcard_featurisation.py)  | running example in paper  | [view](synthesised_code.py#L117) |:white_check_mark:||
| [ldb_featurisation.py](ester/benchmark/ldb_featurisation.py) | [GitHub](https://github.com/LittleDevilBig/Systems-for-AI-Quality/blob/main/main.py) | [view](synthesised_code.py#L164) |:white_check_mark:||
| [titanic_featurisation.py](lester/benchmark/titanic_featurisation.py) | [GitHub](https://github.com/josephmisiti/kaggle-titanic/blob/master/Titanic%20Classification.ipynb) | [view](synthesised_code.py#L206) |:white_check_mark:||

## Model and loss function

| Task with messy original code | Source | Code synthesised by LLM | Correct?| Notes |
|---|---|---|---|---|
| [sklearnlogreg_model.py](lester/benchmark/sklearnlogreg_model.py) | running example in paper | [view](synthesised_code.py#L238) |:white_check_mark:||
| [sklearnsvm_model.py](lester/benchmark/sklearnsvm_model.py)  | [sklearn](https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html) | [view](synthesised_code.py#L257) |:white_check_mark:||
| [sklearnmlp_model.py](lester/benchmark/sklearnmlp_model.py)  | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) | [view](synthesised_code.py#L276)  |:white_check_mark:||





