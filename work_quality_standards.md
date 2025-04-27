WORK QUALITY STANDARDS 

JARGONS USED:
- “Phase” refers to the steps of the pipeline. Examples of phases are: EDA, feature engineering, modelling, dev-ops…
- “Procedure” refers to the smallest unit of work in the pipeline. Several procedures are contained in a phase. Examples of a procedure in feature selection are: multicollinearity, low variance, uniform, boruta, and mutual information.

DEPTH OF REFLECTION
- All parts of the pipelines (e.g: class imbalance) need to discuss all possible methods and alternatives in order to fulfill that procedure . All possible methods/alternatives includes things not seen in class. You need to research those. Set the limit to graduate-level concepts. Using just a single method for a given procedure (maybe the one seen in class) while there may be other meaningful alternatives, will not suffice to have your task marked as successful.
    - Each alternative/method for a same given procedure needs to address the different alternatives tradeoffs and why the final decision for a method for the given procedure was chosen (e.g: why choose to use mutual information over Pearson correlation coefficient).
    - Our work is not covering just what we have seen in class. You have to research external more advanced stuff (Matteo has said this repeatedly) (limit yourself to not get into graduate/PhD-level stuff). If you limit yourself to bare minimum seen in class your contributions aren’t complete.
- Any code that does not have a good explanation contributes 0 to the project and thus it will not be merged. A good explanation explains in complete (mathematical) details how that procedure works (explaining the concept), why it is needed and its (possibles) tradeoffs. This is science all procedures need to be explained with scientific rigor, mathematical formulae and deep explanations on each and every part involved around that procedure. (It is expected to take a lot before u make any decision of what to add to the pipeline, and you need to show ur thinking process)
    - Any claim that a given procedure is benefitting the pipeline in x manner needs to be proven with the corresponding  evidence. Example: SMOTEC will help the pipeline by reducing class imbalance and u show a plot of before and after. 
- Each pipeline (e.g: linear vs tree-based pipelines) is different. It is expected that you apply the corresponding procedure that work best for each pipeline. Trying to overgeneralize a procedure to all the pipelines in order to minimize the work wont get your contributions into the project.

CODE QUALITY
Code that fails to fulfill these requirements wont be integrated into the project
- All functions of the library need to be documented. That is, “docstrings” and “data type annotations” need to be included. 
- No bugs are allowed.
- You have to use the same level of modularity and organization present in current code snippets (each class shouldn’t have more than 300 lines, only one file per class, and each function doesn’t have more than 30 lines, unless it is a plot or other rare cases)
- You have to use the same designs patterns: factory and abstraction function from the base class
- Use descriptive commit names. Write in detail what each commits is adding new to the codebase

TEAM DYNAMICS
- Daily reports on what has been done on that day (tasks fulfilled, errors, things to be done for the next day)
- Daily commits to GitHub 
- When you ask something you have to present what you think the answer to the question is. This answer has to show that you have carefully analyzed the situation. Otherwise it may be showing that you are just simply trying to externalize the thinking process to someone else, which is bad.
- No more than 2 reviews will be done to your work. Otherwise it may be showing that you are just simply trying to externalize the thinking process of your branch to someone else, which is bad. Reviewers (Javi and Cate) are only there for merging your work only. Talk with Cate for merging if your part is research-centric (e.g: Fede), talk with Javi for merging if your part is coding or modelling.
- Before you ask for your code to be merged review this document to make sure you have satisfied all requirements. 

SUMMARY
1. Cover all the space of possible methods/alternatives for a given procedure
2. Apply the best method/alternatives for a given procedure to the corresponding branch (that is, each branch is likely going to require being treated differently)
3. Write reflections of in depth
4. Write good code
5. Communicate more 

WORK TO BE DONE:
⁠- Feature engineering (Fede)
•⁠  ⁠⁠EDA (Cate & Fede)
•⁠  ⁠⁠Data preprocessing (Juan & Irina)
•⁠  ⁠⁠Slides (Juliette)
•⁠  ⁠⁠Neural Nets (at least one feed forward NN) (u assign it)
•⁠  ⁠⁠Extra turilli (at least model calibration and feature importance) (Juan & Irina )
- Prepare domain specific report of questions for Matteo’s friends
•⁠  ⁠⁠Ditch the Jupyter notebook + Make pipeline serializable (Javi) 
With 8 hours per day from Friday 25th to Wednesday 30th you will be able to do this. You would have 40 hours of work, should be more than enough.