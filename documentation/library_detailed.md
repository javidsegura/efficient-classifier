

# LIBRARY DETAILED
We analyze the the library in major detail now. We include folder explanations + some advanced concepts in certain phases (denoted by a .py at the end of the title in the given section). Finally. 
there is a small Q&A at the end for some questions you may have.

## JARGONS USED
Model_sklearn refers to the model you do for instance, model.fit() and model object is the object that captures all the sklearn models accross its diffeernt phases and defines their different behavior.

## library/phases
 - /phases_implementation: implements the code for each phase
 - /runners: different phases runners needed for the pipeline runner

## library/phases/phases_implementation/data_preprocessing/
- Currently has old Javi code. Missing Juan work.

## library/phases/phases_implementation/dataset/
- Loads the dataset and splits it

## library/phases/phases_implementation/dev_ops/
- Currently empty. Slack bot code should go here

## library/phases/phases_implementation/EDA/
- Contains some plots used in model assement

## library/phases/phases_implementation/feature_analysis/
- feature_engineering/ => code with interaction and polynomial effects
- feature_selection/ => contains code for manual and automatic feature selection. Contains a manager too
- feature_transformation/ => allows to do log transformation --currently just implemented for the target variable--, also encoding (both cyclical and categorical). feature_transforamtion_factory.py is the manager.


## library/phases/phases_implementation/modelling/
- result_analysis/ : Contains the code that analyses a given pipelines performance (not the whole pipelines, that is in pipelinemanagers) and saves their performance to disk
- shallow: implementation for shallow (and not deep) models --currently all models are shallow
   - model_definition/: 
     - the different states of the model object
     - the implementations between classification and regression
   - model_optimization/: the different possible optimizers for not-neural nets and the neural network


## library/phases/phases_implementation/modelling/modelling.py
The modelling implementation is complex (lots of different relationships). Even if you are not changing the code in the modelling phase, you will likely need to debug the modelling phase to make sure that prior (e.g., data preprocessing) or post (e.g., model callibration) are working as expected.

A given model has three possible states: pre-tuning (denoted by 'pre'), in-tuning (denoted by 'in'), pre-tuning (denoted by 'post'). \
Phases explanations:
  - pre-tuning: first training of the models with default parameters
  - in-tuning: cv-powered-optimizers tune hyperparameters and constrast results with pre-tuning. Validates the usefulness of hyperparameter opt.
  - post-tuning: after selecting the best model (only one in the not-baseline branch), retrain them and finally check against test set 

Each model behaves differently at each of this phases. Thus each requires a different implementation. Each phase also has its own 'assesment' attribute which stores the predictions, classification metrics, and timeToFit as well as timeToPRedict. Most importantly they also stored the actual model (the one you call model.fit() on) which are used all over the modelling phase.
Assesment has the follow structure 
Assesment currently has the following structure:

```python
- `id`: NoneType
- `timeStamp`: NoneType
- `comments`: NoneType
- `modelName`: str
- `status`: str
- `features_used`: NoneType
- `hyperParameters`: NoneType
- `timeToFit`: float
- `timeToPredict`: float
- `accuracy`: float
- `precision`: float
- `recall`: float
- `f1-score`: float
- `predictions_val`: numpy.ndarray
- `precictions_train`: numpy.ndarray
- `predictions_test`: numpy.ndarray
- `model_sklearn`: sklearn
```


## library/pipeline
 - Coontains the code of pipeline_runner, pipeline_manage, pipeline (object), and the analysis of the pipelines + their serialization

## library/test
 - Contains test for checking the library is working correctly. Currently not implemented.

## library/utils
 - decorators => stores all the decorators used
 - ownModels => we are building on top of scikit-learns interface. Thus, for all the models that we use that are not natively scikit-learn-type need to have their classes adjusted to have the same API interface. We implement that there. Currently two examples of that are: for baseline category, the "majority class predictor" model, and for the non-baseline category, tensforflow's implementation of the feed forward neural network
 - pythonObjects => reimplements a dict that does not allow to add new keys after initialization + contains the save_or_store_plto
 - slackBot => code for communicating with SlackBot

## library/results
- hyperparemter_optimization: stores the NN's optimziation results (for some reason its mandatory to provide a directory to store the results)
- logs: self-explanatory
- model_evaluation: stores the logs of all the models that have been trained since we started using this library. Useful to evaluate the progess
- plots: stores all the images of the plots when uusing the pipeline runner
- serialization stores the objects (models or pipelines) to disk


# Q&A
- Q: How do I add a new metric?
- A: 
  1. Before u store a new metric, you have to mark all the past values as empty. For that use the function in 'library/utils/miscellaneous/write_new_col.py'. 
  2. After this, compute the metric in the 'evaluate' function in the 'classifier.py' file. Add it to the return result of the function with
  the same structure of the "kappan's" metric example
  3. Add the new metric name to configurations.yaml in dataset_runner->metric_to_evaluate.
