
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


from library.utils.phase_runner.phase_runner import PhaseRunner
from library.pipeline.pipeline_manager import PipelineManager

from library.phases.runners.modelling.utils.modelling_runner_states import PreTuningRunner, InTuningRunner, PostTuningRunner

class ModellingRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)
      
      def _model_initializers(self):
            self._create_pipelines_divergences()
            pipelines = list(self.pipeline_manager.pipelines["not_baseline"].values()) # CHANGE TO NN SPECIFIC
            default_pipeline = pipelines[0]

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
                                                                                             FeedForwardNeuralNetwork(num_features=default_pipeline.dataset.X_train.shape[1], 
                                                                                             num_classes=default_pipeline.dataset.y_train.value_counts().shape[0]))
            # Baseline models
            self.pipeline_manager.pipelines["baseline"]["baselines"].modelling.add_model("Logistic Regression (baseline)",
                                                                                             LogisticRegression())
            self.pipeline_manager.pipelines["baseline"]["baselines"].modelling.add_model("Majority Class (baseline)",
                                                                                             MajorityClassClassifier())     
            self._exclude_models()  

      def _exclude_models(self):

            # Ensembled models
            self.pipeline_manager.pipelines["not_baseline"]["ensembled"].modelling.models_to_exclude = ["Gradient Boosting", "Random Forest"]

            # Tree-based models
            self.pipeline_manager.pipelines["not_baseline"]["tree_based"].modelling.models_to_exclude = []

            # Support Vector Machines models
            self.pipeline_manager.pipelines["not_baseline"]["support_vector_machine"].modelling.models_to_exclude = ["Linear Support Vector Machine"]

            # Naive Bayes model
            self.pipeline_manager.pipelines["not_baseline"]["naive_bayes"].modelling.models_to_exclude = []

            # Feed Forward Neural Network model
            self.pipeline_manager.pipelines["not_baseline"]["feed_forward_neural_network"].modelling.models_to_exclude = []

            # Baseline models
            self.pipeline_manager.pipelines["baseline"]["baselines"].modelling.models_to_exclude = ["Logistic Regression (baseline)", ]
      
      def _create_pipelines_divergences(self):
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="ensembled")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="tree_based")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="support_vector_machine")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="naive_bayes")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="feed_forward_neural_network")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="stacking")
            print(self.pipeline_manager.pipelines)
                                                                      
                                                                                             
      def run(self) -> None:
            self._model_initializers()
            
            pre_tuning_runner = PreTuningRunner(self.pipeline_manager)
            pre_tuning_runner.run()

            in_tuning_runner = InTuningRunner(self.pipeline_manager)
            in_tuning_runner.run()

            post_tuning_runner = PostTuningRunner(self.pipeline_manager)
            post_tuning_runner.run()

            return None