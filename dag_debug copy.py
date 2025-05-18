from dag import DAG


# Pipelines
pipelines = [
     {
          "name": "Pipeline1",
          "models": ["Model1", "Model2"]
     },
     {
          "name": "Pipeline2",
          "models": ["Model3"]
     }
]

# Phases
phases = ["Data Preprocessing", "Feature Analysis", "Modelling"]


obj = DAG(pipelines, phases)

obj.add_procedure("Pipeline1", "Data Preprocessing", "Class Imbalance", "no thing was found in here")
obj.add_procedure("Pipeline2", "Data Preprocessing", "No Class Imbalance", "no thing was found in here")



obj.render()