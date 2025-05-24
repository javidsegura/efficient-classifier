# EFFICIENT CLASSIFIER, A DATASET-AGNOSTIC ML CLASSIFER LIBRARY

In this library we present a framework to asses multiple classifiers pipelines for any labelled tabular dataset. We apply this library to the CCCS-CIC-AndMal-2020 dataset. 

## Project Set-up / Tutorial
1. Clone this repository
2. Create a .env with the following credentials (if you want to use the slack bot):
 - SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, SLACK_APP_TOKEN
3. Install libraries from requirements.txt
4. Customize dataset-specific treatment throughout pipeline runners if desired (only if you want to add your own custom logic (e.g, feature engineering, dataset treatment))
5. Customize configurations.yaml with your own preferred settings


## MANTIS: Malware Analysis and Network-based Threat Identification System
As mentioned before, we apply this library to the CCCS-CIC-AndMal-2020 dataset. We conducteed extensive EDA analysis and applied the capture insights for data preprocessing, feature engineering and models optimization.

### Model
Random forest performs the best in pair with stacking model with meta model <i>Logistic Regression</i> and base models <i>Decision Tree, Random Forest, Gaussian Naive Bayes and Feed Forward Neural Network</i>. Them both have a 92% in f1-score. We select random forest as the final model for deployment due to the significant decrease in time for inference and training. 

### Training and tuning
In order to tune the model we used sklearn's bayesian optimzier (keras' implementation for the FFN), tough the library offers the possibilities to use grid search and random forest. In order to adjust the hyperparamter space and the extend of the optimzier search (the number of combinations) please modify modelling_runner->hyperparameters in configurations.yaml

### Inference
Once you have the best model selected out of the assesment after tuning you can serialize your model (either with joblib or pickle). The model serialized contains the results at each stage and the model sklearn object at each stage. In production use the model_sklearn from the assesment attribute in the serialized model. Do model.predict(X) or model.predict_proba(X) if you want soft-predictions.

### Design and Implementation
This library allows for parallel assesment of as many pipelines for different models as provided in configurations.yaml. This library will execute the following phases for each pipeline:

- Dataset load
  - Loads dataset, executes dataset-specific procedures, asses split standard error variation
- Data preprocessing
  - Class imbalance, missing data, outlier detection, bound-checking 
- Feature analysis
  - Feature transformation, feature engineering, feature selection
- Modelling
  - Tuning
  - Post-tuning
      - Callibration (if enabled in configurations.yaml)
- Dev-Ops
  - Send in-real-time logger results to SlackBot (including images results)
  - Store results in CSV
  - Serialize model
  - Pipeline DAG visualization

This library provided us with the scalable framework needed for the complexity of the evaluated pipelines. Through its deeply organized structured, this library allows for short development time and faster debugging. The extensiveness of data stored after a pipeline run allows for efficient auditing as well as a reliable way to revisit the pipeline at retraining time.

### Project Documentation
Dataset information: https://docs.google.com/document/d/1mhRRnYaBdi0t-SR_-fvgbhlN3trc-8hBZ2zbaO6rDUo/edit?usp=sharing

Complete feature context and malware descriptions (In progress): https://docs.google.com/document/d/1yH9gvnJVSH9GLv9ATQ5JQWA2z8Jy4umxxRfMF-y2fiU/edit?usp=sharing

First Iteration Revision: https://docs.google.com/document/d/10R3KM5mM6B9hlxaJ1DYlJZg84UExSFz1kMUTtvPc8fo/edit?usp=sharing

Contributions validation process: https://docs.google.com/spreadsheets/d/1tS9fax-VACRaC3rOBaCBGlC893oibwpqQq_D2NFLN_0/edit?usp=sharing


### Project Status
This project is still under development. It is not guaranteed that this library is free of bugs. It currently has been used only for the aforementioned dataset. We have designed the library to be dataset agnostic, however. We are currently working to make this libray more efficient, more robust as well with added features. 

### Contributing
This library encourages external contributions. Please create your own branch at https://github.com/javidsegura/efficient-classifier and make pull request when your contributions are ready. Please, make sure you provide a documented pull request message detailing your contributions.