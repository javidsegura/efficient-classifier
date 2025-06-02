# Scikit-learn models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, ElasticNet, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

# Self-developed models
from efficient_classifier.utils.ownModels.majorityClassModel import MajorityClassClassifier 
from efficient_classifier.utils.ownModels.neuralNets.feedForward import FeedForwardNeuralNetwork

from efficient_classifier.utils.phase_runner_definition.phase_runner import PhaseRunner
from efficient_classifier.pipeline.pipeline_manager import PipelineManager

from efficient_classifier.phases.runners.modelling.utils.states.modelling_runner_states_pre import PreTuningRunner
from efficient_classifier.phases.runners.modelling.utils.states.modelling_runner_states_in import InTuningRunner
from efficient_classifier.phases.runners.modelling.utils.states.modelling_runner_states_post import PostTuningRunner

class ModellingRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)
      
      def _model_initializers(self):
            """
            We diverge all pipelines first (assuming it has not been done before, delete if it has @Juan or @Fede or @Cate).
            Then add to each independent pipeline the given models. 
            Finally we call the function that excludes all the models that we do not want the training to run (either because we are trying to debug and want to run as fast as possible or
            because we have observed that a certain model is not performing well and taking too long to fit/predict)
            """

            contains_weight = self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["set_weights"]
            
            model_name_to_model_object = {
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "Random Forest": RandomForestClassifier(class_weight=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None),
                        "Decision Tree": DecisionTreeClassifier(class_weight=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None),
                        "Linear SVM": LinearSVC(class_weight=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None),
                        "Non-linear SVM": SVC(class_weight=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None),
                        "Naive Bayes": GaussianNB(),
                        "Logistic Regression": LogisticRegression(class_weight=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None),
                        "Majority Class": MajorityClassClassifier(),
                        "AdaBoost": AdaBoostClassifier(),
                        "XGBoost": XGBClassifier(),
                        "LightGBM": LGBMClassifier(class_weight=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None),
                        "CatBoost": CatBoostClassifier(verbose=False, class_weights=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None),
                        "K-Nearest Neighbors": KNeighborsClassifier(),
                        "Ridge Classifier": RidgeClassifier(class_weight=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None),
                        "Elastic Net": ElasticNet(),
                        "Stochastic Gradient Descent": SGDClassifier(class_weight=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None),
            }



            for pipeline in self.pipeline_manager.variables["general"]["pipelines_names"]["not_baseline"]:
                  if "neural_network" in pipeline:
                        nn_pipeline = self.pipeline_manager.pipelines["not_baseline"][pipeline]
                        model_name_to_model_object["Feed Forward Neural Network"] = FeedForwardNeuralNetwork(
                                                                                          num_features=nn_pipeline.dataset.X_train.shape[1], 
                                                                                          num_classes=nn_pipeline.dataset.y_train.value_counts().shape[0],
                                                                                          batch_size=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["batch_size"],
                                                                                          epochs=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["epochs"],
                                                                                          n_layers=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["n_layers"],
                                                                                          units_per_layer=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["units_per_layer"],
                                                                                          learning_rate=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["learning_rate"],
                                                                                          activations=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["activations"],
                                                                                          kernel_initializer=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["kernel_initializer"],
                                                                                          class_weights=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["class_weights"]["weights"] if contains_weight else None          
                                                                              )
                        break # add all NNs models only once
      
            for category in self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["models_to_include"]:
                  for pipeline in self.pipeline_manager.pipelines[category]:
                        if pipeline == "stacking":
                              continue
                        for model_name in self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["models_to_include"][category][pipeline]:
                              model_name_to_map = model_name
                              if "baseline" in model_name:
                                    model_name_to_map = model_name.replace(" (baseline)", "")
                              if self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["calibration"]["calibrate_models"] and model_name not in self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["calibration"]["not_calibrate_models"]:
                                    print(f"Calibrating {model_name}")
                                    model_sklearn = CalibratedClassifierCV(model_name_to_model_object[model_name_to_map], 
                                                                           method=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["calibration"]["calibration_method"])
                              else:
                                    model_sklearn = model_name_to_model_object[model_name_to_map]

                              self.pipeline_manager.pipelines[category][pipeline].modelling.add_model(
                                    model_name, 
                                    model_sklearn=model_sklearn, 
                                    model_type="neural_network" if "Neural Network" in model_name else "classical") # We handle keras-native model differently 
 
            self._exclude_models()  

      def _exclude_models(self):
            for category in self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["models_to_include"]:
                  for pipeline in self.pipeline_manager.pipelines[category]:
                        if pipeline == "stacking":
                              continue
                        self.pipeline_manager.pipelines[category][pipeline].modelling.models_to_exclude = self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["models_to_exclude"][category][pipeline]

      def start_serialization(self):
            if self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["serialize_models"]["serialize_best_performing_model"]:
                        self.pipeline_manager.serialize_models(models_to_serialize=self.pipeline_manager.best_performing_model["modelName"])
            self.pipeline_manager.serialize_models(models_to_serialize=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["serialize_models"]["models_to_serialize"])
            self.pipeline_manager.serialize_pipelines(pipelines_to_serialize=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["serialize_models"]["pipelines_to_serialize"])
                                                                                             
      def run(self) -> None:
            self._model_initializers()
            
            pre_tuning_runner = PreTuningRunner(self.pipeline_manager,
                                                save_plots=self.include_plots,
                                                save_path=self.save_path)
            pre_results = pre_tuning_runner.run()
            print("-"*30)
            print("STARTING IN TUNING")
            print("-"*30)


            in_tuning_runner = InTuningRunner(self.pipeline_manager,
                                              save_plots=self.include_plots,
                                              save_path=self.save_path)
            in_results = in_tuning_runner.run()
            print("-"*30)
            print("STARTING POST TUNING")
            print("-"*30)

            post_tuning_runner = PostTuningRunner(self.pipeline_manager,
                                                  save_plots=self.include_plots,
                                                  save_path=self.save_path)
            post_results = post_tuning_runner.run()

            self.start_serialization()

            return {"pre_tuning_runner": pre_results,
                    "in_tuning_runner": in_results,
                    "post_tuning_runner": post_results }