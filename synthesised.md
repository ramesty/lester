# Potential of Code Synthesis

At the core of our proposal is the idea to [rewrite messy code](lester/rewrite/__init__.py) for [various pipeline stages](lester/benchmark/__init__.py) based on LLMs with [custom designed prompts](lester/rewrite/prompts.py). 

We provide example rewrites for [nine different rewriting tasks](lester/benchmark) with their corresponding [synthesised pipeline code](synthesised_code.py). These examples have been rewritten with the help of OpenAI's [gpt-4o](https://openai.com/index/hello-gpt-4o/) model and the [generate_dataprep_code](lester/rewrite/__init__.py#L17), [generate_featurisation_code](lester/rewrite/__init__.py#L40) and [generate_model_code](lester/rewrite/__init__.py#L54) methods in our prototype. The corresponding [prompts](lester/rewrite/prompts.py) are available as well.

## Relational data preparation

| Task with messy original code |  Code synthesised by LLM | Source | Notes |
|---|---|---|---|
| [creditcard_dataprep.py](lester/benchmark/creditcard_dataprep.py)  | [view](synthesised_code.py#L2)  | running example in paper ||
| [yichun_dataprep.py](lester/benchmark/yichun_dataprep.py) | [view](synthesised_code.py#L51) | [GitHub](https://github.com/YichunAstrid/e-commerce-use-case/tree/main/1116LogisticRegression) | Manual editing of two lines required<br/> to handle partitioned inputs |
| [amazonreviews_dataprep.py](lester/benchmark/amazonreviews_dataprep.py) | [view](synthesised_code.py#L81) | [GitHub](https://github.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset/blob/master/SVM_model.ipynb) |Dead code generated,<br/> no impact on final output|

## Feature encoding

| Task with messy original code |  Code synthesised by LLM | Source | Notes |
|---|---|---|---|
| [creditcard_featurisation.py](lester/benchmark/creditcard_featurisation.py)  | [view](synthesised_code.py#L117)  | running example in paper ||
| [ldb_featurisation.py](ester/benchmark/ldb_featurisation.py) | [view](synthesised_code.py#L164) | [GitHub](https://github.com/LittleDevilBig/Systems-for-AI-Quality/blob/main/main.py) | |
| [titanic_featurisation.py](lester/benchmark/titanic_featurisation.py) | [view](synthesised_code.py#L206) | [GitHub](https://github.com/josephmisiti/kaggle-titanic/blob/master/Titanic%20Classification.ipynb) ||

## Model and loss function

| Task with messy original code |  Code synthesised by LLM | Source | Notes |
|---|---|---|---|
| [sklearnlogreg_model.py](lester/benchmark/sklearnlogreg_model.py)  | [view](synthesised_code.py#L238)  | running example in paper ||
| [sklearnsvm_model.py](lester/benchmark/sklearnsvm_model.py)  | [view](synthesised_code.py#L257)  | [sklearn](https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html) ||
| [sklearnmlp_model.py](lester/benchmark/sklearnmlp_model.py)  | [view](synthesised_code.py#L276)  | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) ||





