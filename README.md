# Welecome to use ReAD!

ReAD is an Out-of-Distribution(OOD) Detection Method through Relative Activation-Deactivation Abstractions.

Our code is available for reproducible research. This work will be presented soon in ISSRE 2021.

>Zhen Zhang, Peng Wu, Yuhang Chen, Jing Su. Out-of-Distribution Detection through Relative Activation-Deactivation Abstractions, 32nd International Symposium on Software Reliability Engineering (ISSRE 2021), to appear

Our paper reports two types of OOD detection tasks.

Directory ‘ReAD/OOD_detection_task1/' corresponds to the OOD detection tasks of type I, while 
directory ‘ReAD/OOD_detection_task2/' corresponds to the OOD detection tasks of type II. Each subdirectory of them is for a completely independent experiment, and the name of the subdirectory is the name of the training dataset used in the experiment. To run such an experiment, simply execute the following command：

```python
python run_experiment.py
```
For the OOD detection tasks of type II, we reused the model structure settings in the paper 
>Henzinger, T.A., Lukina, A., Schilling, C.: Outside the Box: Abstraction-Based Monitoring of Neural Networks. 24th European Conference on Artificial Intelligence (ECAI) 2020
for the sake of comparison.

The data used in the above experiments are available at:
link：https://pan.baidu.com/s/1gXZy6ID-9Kb-YaSArjwvnw 
password：zzsj 
