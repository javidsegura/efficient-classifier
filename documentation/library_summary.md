

# LIBRARY EXPLANATION

 ### JARGONS USED
 - Pipeline = a collection of similar models being trained from start to finish. All pipelines have the same dataset but not all end up with the same features (tailored for to maximize model performance)
 - Phase = a part in the pipeline (e.g., data preproprecessing, EDA)
 - Procedure = a step within a pipeline (e.g., for feature selection in feature analysis, you have 6 procedures: mutual information, low variance, multicolinearity, PCA, boruta, L1)
 - Subprocedure = provides more orgnaization to the types of procedures. For the above examples you would have two subprocedures: automatic feature selection (boruta, L1) and manual feature selection (mutual information, low variance, multicolinearity, PCA, boruta, L1)
 - Method: an element in the subprocedure array.  For the above example it would be: boruta or L1

 So as a conclusion, a pipeline has a bunch of phases, each phase has at least one procedure, each procedure (usually, but maybe not always) can be categorized in more detail. Then for each of those categories is comprimses of an array of methods. 
 This level of organization its not only a good mental map for the presented methodology but also will be needed for storing the results in the correct directory. 


## INTRODUCTION
Our library contains 4 levels of abastraction, ordered from least to most abstract we have:
 - Phase implementation
   - Represents the lowest-level of implementation. Here you write the functions that directly modify the dataset
 - Pipeline
      - Collects all the phases implementations code and calls them in the proper order
 - Pipeline manager
      - Manages all the pipelines by executing procedures for all the pipelines or specific to only one. In the jup. notebook this is the highest level of abstraction we used.
 - Pipeline runner
      - Controls the pipeline manager by specifiyng the parameters that define the behavior the pipelines. It also is in charge of sending results to the Slack bot and keeping logs of the trainings. It is the highest the level of abstraction. It serves as an orchestrator of the complete training.
      - The pipeline runner is made out of different 

In OOP terms, you will see that pipeline runner initializes, "Pipeline" objects, then passes them to the "pipeline manager" object.

You are asked to write code at the lowest level. Then adequate the behavior per pipeline at the **pipeline manager** and finally integrate it into the the pipeline runner. Sometimes you may not need to write the code at the lowest level, but rather do direct transformation of the dataset at the the pipeline runner. This only is the case when you are making lots of pipeline-specific choices, and thus making a procedure for each does not really make sense. 
 - An example of this is feature engineering. 
 - An example of when this should not happen is feature selection. All pipelines would call the same procedure (lets say mutual information analysis), possibly with different parameters per pipeline, but that is all. In that case what you do is implement at the phase implementation level the code, Pipeline object captures the new code implementation automatically (because Pipeline is always importing the code in phasee implementation), then you use pipeline runner for calling the methods per pipeline or the pipeline manager if you want to execute them all (or want to leave all but a few pipelines outside of that function call).

## YAML
After running the pipeline we have seen how tedious it is to change one parameter that defines the execution of the pipelines at the "Pipeline runner" level. This is because you have navigate through lots of modules (python files). In order to prevent this, we are introducing the use of YAML. YAML (introduced in cloud computing) is commonly used in data-related projects. It provides a way to centralize all the parameters that define the behavior of some script in a single file, thus eliminating the hussle of going file per file chaning everything manually. YAML is a very simple markdown language. You write an element, then subelement (marked by some indentation) and then in the code you just need to call as if it was a dictionary (see the examples in the codebase). 

## SAVING OF RESULTS 
There are three ways in which results are stored: a)logs,  b)images, c) slack-bot. \
Lets start with b):
 - Images are stored by calling the utilities' function "save_or_store_plot". In the pipeline runner this will automatically save the results in a folder, whereas in the jupyter notebook it will just plot it and show it as the cell's output. For the pipeline runner, thus, you need to provide the directory (folder) where you want to store your plot. The pattern is always the following 'save_path + "procedure/sub_procedure/method" (that folder will be created automatically). You then have to pass the filename too.

 c) 
  - The slack-bot sends all the images in results/plots + results/model_evaluation + all the logs.

 a) Logs: 
  - It writes to a logger (a permanent file that is commonly used in software systems to track an execution historial, which is useful for debugging + auditing --in our case just for audting) anything that the runners (present in library/phases/runners) return will be automatically written to the log files + send to the slack bot. This means that not everything you see in the terminal while you execute the script will necessarily be written to the log. Please make sure you only return relevant information. Just keep information that is conclusive of that phase results' and not things that are used for debugging (for that case just write to the terminal with print()). Finally, also make sure you are only sending text to the bot that is not redudant to the images that will also be sent.

## THINGS YOU <i>DONT</i> HAVE TO MODIFY
- Add a new pipeline runner 
- Alter the pipeline manager functionality

## TASKS
 The tasks ahead (note the following list only considers code-related code) can be divided into two groups: a) finishing the pipeline, b) optimizing the pipeline.\
 For a) we have:
  - Integrating data preprocessing (Juan)
  - Integrating feature transformation (Fede)
  - Integrating feature relevance (Irina + Juan)
  - Integrating model callibration (Irina + Juan) 
  - Integrating new activation functions for the NN (Cate)
  - Integrating dataset-specific metrics (Cate)
  - Do residual analysis (Cate) 

For b) we have:
  - Integrating feature engineering (Fede)
  - Changing thresholds for feature selection as needed (Fede)
  - Change hyperpamter optimization grid as needed (Fede)
  - Running everything in a Docker container and sending it to the server (Javi) -- done when all the above tasks are succesfully integrated

## JUPYTER NOTEBOOK
Matteo has indicated that we need to move away from the jupyter notebook, thus the need of the pipeline runner. However, in the meantime of you integrating your code to the pipeline runner, you can debug your code in the jupyter notebook if you feel more comfortable (remember you will be at a lower level of abstraction!)

## PROJECT SET-UP
1. Run pip3 install -r requirements.txt
2. Create a .env at the top level (meaning not inside any folder) and write the credentials share in the gc:
  - SLACK_BOT_TOKEN=xoxb-88...
  - SLACK_SIGNING_SECRET=58...
  - SLACK_APP_TOKEN=xapp-...

There was a bizarre SSL bug with the slack bot in Fede's computer. Try to debug it, otherwise just comment-out the slack-code inside pipeline_runner (just a few lines)

## USEFUL COMMANDS (MacOS)
- CMD + P => lets you access any module (python file) by writing its name. Useful when you know the name and want to avoid navigating all over the library.

## OOP CONCEPTS YOU NEED TO BE FAMILIAR WITH
- What is an object
- What is an attribute
- What is a method
- Class inheritance
- Class abstraction

You need to understand these concepts (less than 45 minutes in total, approx) in order to integrate your code properly.