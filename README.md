# Welecome to use ReAD!

ReAD is an Out-of-Distribution(OOD) Detection Method through Relative Activation-Deactivation Abstractions.

Our code is available for reproducible research. The study appears in the ISSRE 2021(paper link).

Our paper shows two types of OOD detection task.

Directory ‘ReAD/OOD_detection_task1/' corresponds to the OOD Detection of type I, Each subdirectory is a complete independent experiment, and the name of the directory means the name of the training data set of the model. If you want to reproduce the experimental results in the paper, you only need to execute the command：

```python
python run_experiment.py
```

Directory ‘ReAD/OOD_detection_task2/' corresponds to the OOD Detection of type II, it's execution is completely similar to tpye I. In this task, the model structure setting is relatively simple and the test accuracy of model is not very good, but in order to compare with work Outside-the-Box[paper_link], we completely adopted the model structure setting in their article.