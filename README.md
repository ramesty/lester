## Overview

This repository contains the supplementary material for our submission _"Towards Regaining Control over
Messy Machine Learning Pipelines"_. Our prototype runs [declaratively specified ML pipelines](lester/classification.py) via a [dataframe API](lester/__init__.py) with basic relational operations and provenance tracking for rows and columns, estimator/transformers with [matrix column provenance](lester/feature_provenance.py) and models specified in PyTorch.

### LLM-assisted rewrites of messy pipeline code

At the core of our proposal is the idea to [rewrite messy code](lester/rewrite/__init__.py) for [various pipeline stages](lester/benchmark/__init__.py) based on LLMs with [custom designed prompts](lester/rewrite/prompts.py). We provide example rewrites for [nine different rewriting tasks](lester/benchmark) with their corresponding [synthesise pipeline code](synthesised_code.py).

### Provenance-based unlearning for all pipeline artifacts

Our prototype materialises the produced [pipeline artifacts](lester/unlearning/artifacts.py) and subsequently allows us to conduct provenance-based unlearning on these artifacts with low latency. We implement the unlearning of [feature values](lester/unlearning/feature_deletion.py) and [instances](lester/unlearning/instance_deletion.py) from all pipeline artifacts. The deletion for the relational training data and the encoded features is based on provenance and uses dataframe and numpy operations. The unlearning from the trained model is conducted via a recently proposed [first-order update](https://www.ndss-symposium.org/wp-content/uploads/2023/02/ndss2023_s87_paper.pdf).

## Running example

 * [Messy original pipeline](messy_original_pipeline.py) for our running example
 * Using OpenAI's [gpt-4o](https://openai.com/index/hello-gpt-4o/) model via the langchain API
 *  [Synthesised data preparation code](synthesised_code.py#L2) from [messy input code](lester/benchmark/creditcard_dataprep.py)  via the [generate_dataprep_code](lester/rewrite/__init__.py#L17)
 *  [Synthesised featurisation code](synthesised_code.py#L117) from [messy input code](lester/benchmark/creditcard_featurisation.py) via the [generate_featurisation_code](lester/rewrite/__init__.py#L40)
 *  [Synthesised learning code](synthesised_code.py#L117) from [messy input code](lester/benchmark/sklearnlogreg_model.py) via the  [generate_model_code](lester/rewrite/__init__.py#L54)

![example code transformation](running-example-rewritten.jpg)

## Experiments

* [evaluation of the synthesised code](experiment__rewrite.py) for the nine example tasks
* [retraining from scratch](experiment__retraining_time.py) vs [targeted unlearning](experiment__unlearning.py). Pipeline must be [executed first](run_rewritten_pipeline.py) as preparation. +link to large file
  
