# ML_Feature_Selector_Tool
Machine Learning Feature Selector Tool is a small script intended 
to help anyone working with machine learning, choose or target 
features of a data set by measuring the correlation of each 
feature with the target prediction.

## Description
Machine Learning Feature Selector Tool is a small script intended 
to help anyone working with machine learning, choose or target 
features of a data set by measuring the correlation of each 
feature with the target prediction. To perform this task
the python script defines 6 different ways to measure the 
correlation between each feature and the target y variable.
The correlation measures used in this script are the following:
1. Pearson
2. Chi-square
3. Recursive Feature Elimination
4. Embedded logistic Regression
5. Embedded Random Forest
6. Embedded light GBM

Each correlation measure will choose it's own n features as the
features that correlate the most to the target y variable.
Afterwards the features selected by each individual method
to measure correlation are compared by "voting". So if one
specific feature is selected by all 6 methods, that feature
will have 6 votes.

The user can define how much n features each method will select
and the amount of votes a feature needs to be selected as a 
"top" feature.

Feature selector scripts can not handle categorical features
so it expects only numerical columns.

## Get Started
### Dependencies
libraries used in the script are listed in the following
1. pandas==1.3.5
2. lightgbm==4.1.0
3. numpy==1.21.6
4. scikit-learn==1.0.2

## Executing Example
1. Download FeatureSelectorTool.py
2. Download example csv fifa_19_FeatureSelector.csv
3. execute FeatureSelectorTool.py using python via your cmd
4. Follow instructions

## Help
Check the pre processing notebook example to see a concrete example on how to
prepare your own files to use them with FeatureSelectorTool.py

## Authors
Juan Henao Barrios

## Acknowledgments
Special thanks to Vejey Gandyer who is always willing to help/teach new things.
