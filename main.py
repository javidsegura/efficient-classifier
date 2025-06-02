from efficient_classifier.root import run_pipeline
import yaml

PATH = "configurations.yaml"

variables = yaml.load(open(PATH), Loader=yaml.FullLoader)

if __name__ == "__main__":
      run_pipeline(variables=variables)