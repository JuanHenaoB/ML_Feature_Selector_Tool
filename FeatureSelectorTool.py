# Python Script for an Auto Feature Selection tool
# Import libraries
import numpy as np
import pandas as pd

############################
######Define functions######
############################

########################################
#Pearson Correlation selector function
def cor_selector(X, y,num_feats):
    cor_list = [] #create an empty list to get the correlations from X features with y
    feature_list = list(X.columns)#create a feature list
    for feature in feature_list:#For all features
        cor_list.append(np.corrcoef(X[feature], y, rowvar=False)[0,1])#calculate correlation between feature and y
    #replace NaNs with 0 in the correlation list
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    #Organize features, and select the ones that have the top n correlation to y
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    #Create a list of boolean as a support to select features
    cor_support = [True if i in cor_feature else False for i in feature_list]
    return cor_support, cor_feature

#######################################
#Chi 2 Selector function
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

def chi_squared_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X) # Normalize Data
    #Create an object that calculates the chi2 score and selects top 30 features
    chi_selector = SelectKBest(score_func=chi2, k=num_feats) 
    chi_selector.fit(X_norm, y)#Run score functions 
    chi_support = chi_selector.get_support() #Get a mask, or integer index, of the features selected.
    chi_feature = X.loc[:,chi_support].columns.tolist() # get the selected features
    return chi_support, chi_feature

######################################################
#Recursive Feature Elimination (RFE) Selector Function 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def rfe_selector(X, y, num_feats):
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats,
                   step=10, verbose=10)
    rfe_selector.fit(MinMaxScaler().fit_transform(X), y)
    rfe_support = rfe_selector.get_support() #Get a mask, or integer index, of the features selected.
    rfe_feature = X.loc[:,rfe_support].columns.tolist() # Get list of selected features
    return rfe_support, rfe_feature

##################################################
#Logistic Regression Selector Model
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def embedded_log_reg_selector(X, y, num_feats):
    #I'm using l2 penalty because skleark prompts l1 penalty was deprecated
    embeded_lr_selector = SelectFromModel(estimator=LogisticRegression(penalty="l2"), max_features=num_feats)
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_lr_selector.fit(X_norm, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    return embeded_lr_support, embeded_lr_feature

###################################################
#Random Forest Selector Model
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def embedded_rf_selector(X, y, num_feats):
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X,y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    return embeded_rf_support, embeded_rf_feature

##########################################################
#light GBM Selector Model
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from lightgbm import early_stopping

def embedded_lgbm_selector(X, y, num_feats):
    import re
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2, reg_alpha=3, 
                          reg_lambda=1, min_split_gain=0.01, min_child_weight=40 )
    embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
    embeded_lgb_selector.fit(X,y)
    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
    return embeded_lgb_support, embeded_lgb_feature

############################################################
#Data pre-processing for Script
def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    # Since generalizing a Data pre processing for any dataset is unpractical the script will expect a pre processed dataset
    #And this step will consist only on separating X and y values
    df = pd.read_csv(dataset_path)
    X = df.iloc[:,0:-1]
    feature_name = X.columns.tolist()
    y = df.iloc[:,-1]
    # Your code ends here
    return X, y, feature_name

##########################################################
#AutoFeature Selection function
def autoFeatureSelector(dataset_path, methods=[], num_feats=30, top_votes=3):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    #pearson, chi-square, Recursive Feature Elimination, Embbeded Logistic Regression, 
    #Embbeded Random Forest, Embbeded light GBM
    
    # preprocessing
    X, y, feature_name = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embeded_lr_support, embeded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embeded_rf_support, embeded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embeded_lgb_support, embeded_lgb_feature = embedded_lgbm_selector(X, y, num_feats)
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 
                                         'RFE':rfe_support, 'Logistics':embeded_lr_support,'Random Forest':embeded_rf_support, 
                                         'LightGBM':embeded_lgb_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top n features
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    best_voted_features = feature_selection_df.loc[feature_selection_df.Total >= top_votes]
    selected_features = feature_selection_df.head(num_feats)
    return best_voted_features, selected_features

#########################################
#main function
if __name__ == '__main__':
    print(" \n #################################################################### \n")
    print(" \n Welcome to the Feature selector tool, please follow the instructions \n")
    print(" \n #################################################################### \n")
    
    print("Feature Selctor tool will use 6 different methods to measure the correlation of each Feature(column)")
    print("and the variable to predict(last column). The methods this tool uses are the following 6 : \n")
    print("\n Pearson,\n chi-square,\n Recursive Feature Elimination,\n Embbeded Logistic Regression,\n Embbeded Random Forest,\n Embbeded light GBM \n")
    print("\n Each method will select it's own top_n_features, methods will vote on their top features")
    print(" This means that the maximum amount of votes one feature can get is 6")
    print(" That way we'll have top_m_voted_features where m < n \n")
    print("Please note that this script expects a data-set where the last column is the target variable "+"y"+" to predict")
    print("and all the other columns are scaled/encoded/preprocessed features \n" )
    path = str(input("Please provide the path to your pre-processed Data set \n"))
    number_feats = int(input("\nPlease input the number of maximum n features you wish to select \n"))
    top_feats = int(input("\nPlease input the number of m votes for top features \n"))
    print("\n")
    best_voted_features, selected_features = autoFeatureSelector(path, methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'], num_feats=number_feats, top_votes=top_feats)
    print("\n top "+str(number_feats)+" selected features are: \n", selected_features)
    print("\n top features with "+str(top_feats)+" votes are: \n", best_voted_features)
