
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
            nn_pipeline = self.pipeline_manager.pipelines["not_baseline"]["feed_forward_neural_network"]

            model_name_to_model_object = {
                  "not_baseline": {
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "Random Forest": RandomForestClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Linear Support Vector Machine": LinearSVC(),
                        "Non-linear Support Vector Machine": SVC(),
                        "Naive Bayes": GaussianNB(),
                        "Feed Forward Neural Network": FeedForwardNeuralNetwork(
                                                                                          num_features=nn_pipeline.dataset.X_train.shape[1], 
                                                                                          num_classes=nn_pipeline.dataset.y_train.value_counts().shape[0],
                                                                                          batch_size=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["batch_size"],
                                                                                          epochs=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["epochs"],
                                                                                          n_layers=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["n_layers"],
                                                                                          units_per_layer=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["units_per_layer"],
                                                                                          learning_rate=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["learning_rate"],
                                                                                          activations=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["activations"],
                                                                                          kernel_initializer=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["neural_network"]["initial_architecture"]["kernel_initializer"]
                                                                              ),
                  },
                  "baseline": {
                        "Logistic Regression (baseline)": LogisticRegression(),
                        "Majority Class (baseline)": MajorityClassClassifier(),
                  }
            }
      
            for category in self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["models_to_include"]:
                  for pipeline in self.pipeline_manager.pipelines[category]:
                        if pipeline == "stacking":
                              continue
                        for model_name in self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["models_to_include"][category][pipeline]:
                              self.pipeline_manager.pipelines[category][pipeline].modelling.add_model(
                                    model_name, 
                                    model_name_to_model_object[category][model_name], 
                                    model_type="neural_network" if model_name == "Feed Forward Neural Network" else "classical") # We handle keras-native model differently 
 
            self._exclude_models()  

      def _exclude_models(self):
            for category in self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["models_to_include"]:
                  for pipeline in self.pipeline_manager.pipelines[category]:
                        if pipeline == "stacking":
                              continue
                        self.pipeline_manager.pipelines[category][pipeline].modelling.models_to_exclude = self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["models_to_exclude"][category][pipeline]
                                                                                              
                                                                                             
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

            if self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["serialize_models"]["serialize_best_performing_model"]:
                        self.pipeline_manager.serialize_models(models_to_serialize=self.pipeline_manager.best_performing_model["modelName"])
            self.pipeline_manager.serialize_models(models_to_serialize=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["serialize_models"]["models_to_serialize"])
            self.pipeline_manager.serialize_pipelines(pipelines_to_serialize=self.pipeline_manager.variables["phase_runners"]["modelling_runner"]["serialize_models"]["pipelines_to_serialize"])

            return {"pre_tuning_runner": pre_results,
                    "in_tuning_runner": None,
                    "post_tuning_runner": None}