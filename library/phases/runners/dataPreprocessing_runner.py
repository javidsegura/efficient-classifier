from library.utils.phase_runner_definition.phase_runner import PhaseRunner
from library.pipeline.pipeline_manager import PipelineManager
from library.phases.phases_implementation.data_preprocessing.data_preprocessing import Preprocessing
import yaml
from pathlib import Path

class DataPreprocessingRunner(PhaseRunner):
      def __init__(self, pipeline_manager: PipelineManager, include_plots: bool = False, save_path: str = "") -> None:
            super().__init__(pipeline_manager, include_plots, save_path)
            config_path = Path(__file__).parent / 'preprocessing_config.yaml'
            with open(config_path, 'r') as f:
                  self.config = yaml.safe_load(f)
            
      def _create_pipelines_divergences(self):
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="ensembled")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="tree_based")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="support_vector_machine")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="naive_bayes")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="feed_forward_neural_network")
            self.pipeline_manager.create_pipeline_divergence(category="not_baseline", pipelineName="stacking")
            self.pipeline_manager.create_pipeline_divergence(category="baseline", pipelineName="baselines")
            print(f"Pipelines AFTER divergences: {self.pipeline_manager.pipelines}")
            
      
      def _preprocessor_initializers(self):
            # Ensembled Pipelines
            self.pipeline_manager.pipelines["not_baseline"]["ensembled"].preprocessing
            
      def _execute_preprocessing(self, preprocessing: Preprocessing, cfg: dict):
            # 1. Missing Values and Duplicate Analysis
            print("Preprocessing --- Missing Values & Duplicates")
            print("-"*15)
            preprocessing.uncomplete_data_obj.get_missing_values()
            preprocessing.uncomplete_data_obj.analyze_duplicates()
            
            # 2. Outliers & Bounds
            print("Preprocessing --- Bounds & Outliers")
            print("-"*15)
            preprocessing.outliers_bounds_obj.get_outliers()
            preprocessing.outliers_bounds_obj.bound_checking()
            
            # 3. Feature Scaling
            print("Preprocessing --- Feature Scaling")
            print("-"*15)
            preprocessing.feature_scaling_obj.scale_features()
            
            # 4. Class Imbalance
            print("Preprocessing --- Class Imbalance")
            print("-"*15)
            preprocessing.class_imbalance_obj.class_imbalance()
        
      def _execute_preprocessing_with_config(self, preprocessing: Preprocessing, cfg: dict) -> None:
            """
            Apply missing-value handling, duplicate analysis, outlier bounding,
            feature scaling, and class-imbalance correction based on cfg,
            returning a composed summary of operations/results.
            """
            messages = []

            # 1) Missing values & Duplicate analysis
            print("\nPreprocessing --- Missing Values & Duplicates\n")
            missing_cfg = cfg.get('missing', {})
            missing_res = preprocessing.uncomplete_data_obj.get_missing_values(
                  placeholders=missing_cfg.get('placeholders'),
                  plot=missing_cfg.get('plot', False)
            )
            messages.append(f"Handled missing values (plot={missing_cfg.get('plot', False)}): {missing_res}")

            dup_cfg = cfg.get('duplicates', {})
            if dup_cfg.get('analyze', False):
                  dup_res = preprocessing.uncomplete_data_obj.analyze_duplicates(
                  plot=dup_cfg.get('plot', False)
                  )
                  messages.append(f"Duplicates analyzed (plot={dup_cfg.get('plot', False)}): {dup_res}")
            else:
                  messages.append("Skipped duplicate analysis")

            # 2) Outlier detection & bounding
            print("\nPreprocessing --- Bounds & Outliers\n")
            out_cfg = cfg.get('outliers', {})
            out_res = preprocessing.outliers_bounds_obj.get_outliers(
                  detection_type=out_cfg.get('detection_type'),
                  plot=out_cfg.get('plot', False)
            )
            preprocessing.outliers_bounds_obj.bound_checking()
            messages.append(f"Outliers detected by {out_cfg.get('detection_type')} (plot={out_cfg.get('plot', False)}): {out_res}")

            # 3) Feature scaling
            print("\nPreprocessing --- Feature Scaling\n")
            scale_cfg = cfg.get('scaling', {})
            scale_res = preprocessing.feature_scaling_obj.scale_features(
                  scaler=scale_cfg.get('scaler'),
                  columnsToScale=preprocessing.dataset.X_train.select_dtypes(include=["number"]).columns,
                  plot=scale_cfg.get('plot', False)
            )
            messages.append(f"Features scaled with {scale_cfg.get('scaler')} (plot={scale_cfg.get('plot', False)}): {scale_res}")

            # 4) Class imbalance correction
            print("\nPreprocessing --- Class Imbalance\n")
            imb_cfg = cfg.get('imbalance', {})
            if imb_cfg.get('perform', False):
                  imb_res = preprocessing.class_imbalance_obj.class_imbalance(plot=imb_cfg.get('plot', False))
                  messages.append(f"Class imbalance correction performed (plot={imb_cfg.get('plot', False)}): {imb_res}")
            else:
                  messages.append("Skipped class imbalance correction (perform=False)")

            # Combine all messages into one summary
            return "; ".join(messages)
    
      
      
      def run(self) -> None:
            print(self.pipeline_manager.pipelines)
            self._create_pipelines_divergences()
            print(self.pipeline_manager.pipelines)
            print("-"*30)
            print("STARTING PREPROCESSING")
            print("-"*30)
            
            
            results = {}

            for category_name, pipelines in self.pipeline_manager.pipelines.items():
                  results.setdefault(category_name, {})  
                  for pipeline_name, pipeline in pipelines.items():
                        print(f"--> Running preprocessing on pipeline: {category_name} / {pipeline_name}")
                        print("-"*30)
                        
                        # Lookup config for this pipeline
                        cfg = self.config.get(pipeline_name)
                        if cfg is None:
                              raise KeyError(f"No preprocessing config found for pipeline '{pipeline_name}'")

                        # Create a fresh Preprocessing instance with the current dataset
                        preprocessor = Preprocessing(dataset=pipeline.dataset)
                        
                        try:
                              summary = self._execute_preprocessing_with_config(preprocessing=preprocessor, cfg=cfg)
                              print(summary)
                              results[category_name][pipeline_name] = summary
                        except Exception as e:
                              error_msg = f"Failed preprocessing on {pipeline_name}: {e}"
                              print(error_msg)
                              results[category_name][pipeline_name] = error_msg
            return results