
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
from library.utils.ownModels.majorityClassModel import MajorityClassClassifier 
from library.utils.ownModels.neuralNets.feedForward import FeedForwardNeuralNetwork


from library.utils.phase_runner_definition.phase_runner import PhaseRunner
from library.pipeline.pipeline_manager import PipelineManager

from library.phases.runners.modelling.utils.states.modelling_runner_states_pre import PreTuningRunner
from library.phases.runners.modelling.utils.states.modelling_runner_states_in import InTuningRunner
from library.phases.runners.modelling.utils.states.modelling_runner_states_post import PostTuningRunner

class ModellingRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "", serialize_results: bool = False) -> None:
            super().__init__(pipeline_manager, include_plots, save_path)
            self.serialize_results = serialize_results
      
      def _model_initializers(self):
            """
            We diverge all pipelines first (assuming it has not been done before, delete if it has @Juan or @Fede or @Cate).
            Then add to each independent pipeline the given models. 
            Finally we call the function that excludes all the models that we do not want the training to run (either because we are trying to debug and want to run as fast as possible or
            because we have observed that a certain model is not performing well and taking too long to fit/predict)
            """
            #self._create_pipelines_divergences()
            nn_pipeline = self.pipeline_manager.pipelines["not_baseline"]["feed_forward_neural_network"]

            # Ensembled models
            self.pipeline_manager.pipelines["not_baseline"]["ensembled"].modelling.add_model("Gradient Boosting", 
                                                                                             GradientBoostingClassifier())
            self.pipeline_manager.pipelines["not_baseline"]["ensembled"].modelling.add_model("Random Forest",
                                                                                             RandomForestClassifier())
            # Tree-based models
            self.pipeline_manager.pipelines["not_baseline"]["tree_based"].modelling.add_model("Decision Tree",
                                                                                             DecisionTreeClassifier())
            # Support Vector Machines models
            self.pipeline_manager.pipelines["not_baseline"]["support_vector_machine"].modelling.add_model("Non-linear Support Vector Machine",
                                                                                             SVC())
            self.pipeline_manager.pipelines["not_baseline"]["support_vector_machine"].modelling.add_model("Linear Support Vector Machine",
                                                                                             LinearSVC())
            # Naive Bayes model
            self.pipeline_manager.pipelines["not_baseline"]["naive_bayes"].modelling.add_model("Naive Bayes",
                                                                                             GaussianNB())
            # Neural Network model
            self.pipeline_manager.pipelines["not_baseline"]["feed_forward_neural_network"].modelling.add_model("Feed Forward Neural Network",
                                                                                             FeedForwardNeuralNetwork(
                                                                                                num_features=nn_pipeline.dataset.X_train.shape[1], 
                                                                                                num_classes=nn_pipeline.dataset.y_train.value_counts().shape[0],
                                                                                                batch_size=self.pipeline_manager.variables["modelling_runner"]["neural_network"]["initial_architecture"]["batch_size"],
                                                                                                epochs=self.pipeline_manager.variables["modelling_runner"]["neural_network"]["initial_architecture"]["epochs"],
                                                                                                n_layers=self.pipeline_manager.variables["modelling_runner"]["neural_network"]["initial_architecture"]["n_layers"],
                                                                                                units_per_layer=self.pipeline_manager.variables["modelling_runner"]["neural_network"]["initial_architecture"]["units_per_layer"],
                                                                                                activations=self.pipeline_manager.variables["modelling_runner"]["neural_network"]["initial_architecture"]["activations"],
                                                                                                learning_rate=self.pipeline_manager.variables["modelling_runner"]["neural_network"]["initial_architecture"]["learning_rate"]
                                                                                                ),
                                                                                             model_type="neural_network")
            # Baseline models
            self.pipeline_manager.pipelines["baseline"]["baselines"].modelling.add_model("Logistic Regression (baseline)",
                                                                                             LogisticRegression(max_iter=1000))
            self.pipeline_manager.pipelines["baseline"]["baselines"].modelling.add_model("Majority Class (baseline)",
                                                                                             MajorityClassClassifier())     
            self._exclude_models()  

      def _exclude_models(self):

            # Ensembled models
            self.pipeline_manager.pipelines["not_baseline"]["ensembled"].modelling.models_to_exclude = self.pipeline_manager.variables["modelling_runner"]["models_to_exclude"]["ensembled"]

            # Tree-based models
            self.pipeline_manager.pipelines["not_baseline"]["tree_based"].modelling.models_to_exclude = self.pipeline_manager.variables["modelling_runner"]["models_to_exclude"]["tree_based"]

            # Support Vector Machines models
            self.pipeline_manager.pipelines["not_baseline"]["support_vector_machine"].modelling.models_to_exclude = self.pipeline_manager.variables["modelling_runner"]["models_to_exclude"]["support_vector_machine"]

            # Naive Bayes model
            self.pipeline_manager.pipelines["not_baseline"]["naive_bayes"].modelling.models_to_exclude = self.pipeline_manager.variables["modelling_runner"]["models_to_exclude"]["naive_bayes"]

            # Feed Forward Neural Network model
            self.pipeline_manager.pipelines["not_baseline"]["feed_forward_neural_network"].modelling.models_to_exclude = self.pipeline_manager.variables["modelling_runner"]["models_to_exclude"]["feed_forward_neural_network"]

            # Baseline models
            self.pipeline_manager.pipelines["baseline"]["baselines"].modelling.models_to_exclude = self.pipeline_manager.variables["modelling_runner"]["models_to_exclude"]["baselines"]
      
      def _create_pipelines_divergences(self):
            # self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="ensembled")
            # self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="tree_based")
            # self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="support_vector_machine")
            # self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="naive_bayes")
            # self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="feed_forward_neural_network")
            # self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="stacking")
            # self.pipeline_manager.create_pipeline_divergence(category="baseline", pipelineName="baselines")
            # print(f"Pipelines AFTER divergences: {self.pipeline_manager.pipelines}")
            pass
                                                                      
                                                                                             
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

            if self.serialize_results:
                  if self.pipeline_manager.variables["modelling_runner"]["serialize_models"]["serialize_best_performing_model"]:
                        self.pipeline_manager.serialize_models(models_to_serialize=self.pipeline_manager.best_performing_model["modelName"])
                  self.pipeline_manager.serialize_models(models_to_serialize=self.pipeline_manager.variables["modelling_runner"]["serialize_models"]["models_to_serialize"])
                  self.pipeline_manager.serialize_pipelines(pipelines_to_serialize=self.pipeline_manager.variables["modelling_runner"]["serialize_models"]["pipelines_to_serialize"])

            return {"pre_tuning_runner": pre_results,
                    "in_tuning_runner": in_results,
                    "post_tuning_runner": post_results}