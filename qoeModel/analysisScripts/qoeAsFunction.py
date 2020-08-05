import json
import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, Ridge
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_regression, mutual_info_regression, RFECV
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, make_scorer
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.neural_network import MLPRegressor
from sklearn.utils import resample
from sklearn.utils.validation import check_X_y
from statistics import mode, mean
from sklearn import base, metrics
from sklearn.base import clone
from mord import LogisticAT, OrdinalRidge, LAD
from sklearn.tree import DecisionTreeRegressor
from functools import reduce
from itertools import chain
from collections import Counter
from scipy import optimize
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.dummy import DummyRegressor

pp = pprint.PrettyPrinter(indent=4)  # Pretty Printers
pd.options.mode.chained_assignment = None  # Silence warnings
matplotlib.rcParams.update({'errorbar.capsize': 2})

# Separate features from labels from training/test set


def splitData(df, test_size=0.2):
    # Split to 80 20 -- use CV for train split
    train, test = train_test_split(df, test_size=test_size, shuffle=True)
    X = train.iloc[:, :-1]
    y = train.iloc[:, -1]
    test_features = test.iloc[:, :-1]
    test_labels = test.iloc[:, -1]
    return X, y, test_features, test_labels

# Create a scorer for classification using regression models


def acc_fun(target_true, target_fit):
    target_fit = np.round(target_fit)
    target_fit.astype('int')
    return accuracy_score(target_true, target_fit)


################################# - START - #################################
############################# - PREPROCESSING - #############################
# Read the json file
with open("youtube/CABaRet_Analysis/qoeModel/ratings_1-0.json") as f:
    data = json.load(f)
# with open("youtube/CABaRet_Analysis/qoeModel/updatedDatasetWithPrevQoR.json") as f:
#     data = json.load(f)

# Extract current columns
df_base = pd.DataFrame(data, columns=["QoS", "QoR", "Int", "QoE", "Hit"])

# Remove Cache Hit - not necessary for QoE = f(QoR, Int, QoS)
df = df_base.iloc[:, :-1]
df.pop('QoR')
# Hold the class variable to append it after adding more features in the feature space
# df = df_base
qoeColumn = df.pop('QoE')
# Add Feature Space
df['minIntQoS'] = np.minimum(df['Int'], df['QoS'])
# df['minIntQoR'] = np.minimum(df['Int'], df['QoR'])
# df['minQoSQoR'] = np.minimum(df['QoS'], df['QoR'])
# df['maxIntQoS'] = np.maximum(df['Int'], df['QoS'])
# df['maxIntQoR'] = np.maximum(df['Int'], df['QoR'])
# df['maxQoSQoR'] = np.maximum(df['QoS'], df['QoR'])
# df['expQoS'] = np.exp(df['QoS'])
# df['expQoR'] = np.exp(df['QoR'])
# df['expInt'] = np.exp(df['Int'])
# df['logQoS'] = np.log(df['QoS'])
# df['logQoR'] = np.log(df['QoR'])
# df['logInt'] = np.log(df['Int'])
# df['int:qos'] = df['Int']/df['QoS']
# df['int:qor'] = df['Int']/df['QoR']
# df['qor:qos'] = df['QoR']/df['QoS']
# df['qor:int'] = df['QoR']/df['Int']
# df['qos:int'] = df['QoS']/df['Int']
# df['qos:qor'] = df['QoS']/df['QoR']
# df['int*qos'] = df['Int']*df['QoS']
# df['int*qor'] = df['Int']*df['QoR']
# df['qor*qos'] = df['QoR']*df['QoS']
# df.pop('QoR')
# Append Class Variable in the end
df['QoE'] = qoeColumn

# Remove outliers
ind = np.where((np.maximum(df['QoS'].values, df['Int'].values) >= df["QoE"].values) &
               (np.minimum(df['QoS'].values, df['Int'].values) <= df["QoE"].values))
df = df.loc[ind]
print(df)

# Calculate pvalues of features using chi2
# This is a preprocessing step to find important features for ml models
chi2, pval = chi2(df.iloc[:, :-1], df["QoE"])
zippedPvaluesRaw = list(zip(df.columns, pval))
zippedPvalues = list(zip(df.columns, pval < 0.05))
importantFeaturesTuples = [t for t in zippedPvalues if t[1] == True]
importantFeatures = list(f[0] for f in importantFeaturesTuples)
print("Important features using pvalue, chi2:")
pp.pprint(importantFeatures)
updatedDataset = df[importantFeatures]
updatedDataset['QoE'] = df['QoE']

# # Print correlation matrix among features
print("Correlation of initial features:")
print(updatedDataset.corr('pearson'))
############################# - PREPROCESSING - #############################
################################## - END - ##################################


################################ - START - #################################
############################# - CRAFT MODELS - #############################
# Models to Train
ml_models = [\
    # LogisticRegression().fit(X, y),\
    # LinearRegression(fit_intercept=True, normalize=True),\
    # LogisticRegression(multi_class='ovr',
    #                    solver='liblinear',
    #                    class_weight='balanced'),\
    # LogisticRegression(multi_class='multinomial',
    #                    solver='lbfgs',
    #                    class_weight='balanced', max_iter=10000),\
    # LogisticRegression(solver='liblinear', multi_class='auto', penalty='l1'),\
    # LogisticRegression(solver='newton-cg',
    #                    multi_class='multinomial', penalty='l2'),\
    # LogisticRegression(n_jobs=-1, solver='saga',
    #                    multi_class='multinomial', max_iter=2000),\
    # LogisticRegression(
    #     multi_class='multinomial', solver='newton-cg', fit_intercept=True),\
    # RandomForestClassifier(),\
    # RandomForestClassifier(class_weight='balanced', n_estimators=100,
    #                        oob_score=True, random_state=123),\
    # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                        max_depth=2, max_features='auto', max_leaf_nodes=None,
    #                        min_impurity_decrease=0.0, min_impurity_split=None,
    #                        min_samples_leaf=1, min_samples_split=2,
    #                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
    #                        oob_score=False, random_state=0, verbose=0, warm_start=False),\
    # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
    #                        max_depth=2, max_features='auto', max_leaf_nodes=None,
    #                        min_impurity_decrease=0.0, min_impurity_split=None,
    #                        min_samples_leaf=1, min_samples_split=2,
    #                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
    #                        oob_score=False, random_state=0, verbose=0, warm_start=False),\
    # SVC(kernel='rbf', gamma="auto"),\
    # SVC(kernel='poly', gamma="auto"),\
    # SVC(kernel='sigmoid', gamma="auto"),\
    # SVR(gamma="auto"),\
    # LAD(epsilon=0),\
    # LogisticAT(alpha=0.00),\
    # LogisticAT(alpha=1.00),\
    # LogisticAT(alpha=2.00),\
    # LogisticAT(alpha=5.00),\
    # LogisticAT(alpha=10.00),\
    # LogisticAT(alpha=20.00),\
    # LogisticAT(alpha=100.00),\
    # LogisticAT(alpha=200.00),\
    # LogisticAT(alpha=1000.00)
    LogisticRegression(multi_class='multinomial',
                       solver='lbfgs',
                       class_weight='balanced', max_iter=10000),\
    # RandomForestClassifier(class_weight='balanced', n_estimators=100,
    #                        oob_score=True, random_state=123),\
    DecisionTreeRegressor(),
    # SVC(kernel='rbf', gamma="auto"),\
    SVR(gamma="auto"),\
    MLPRegressor(solver='lbfgs', alpha=1e-5,
                 hidden_layer_sizes=(16, 16), random_state=1),\
    Ridge(alpha=1.0),\
    LogisticAT(alpha=0.00),\
    # DummyRegressor(strategy="constant", constant=3)
    # OrdinalRidge(alpha=0),\
    # LinearSVC(C=0.01, max_iter=100000, penalty="l1", dual=False)
]

############################## - CRAFT MODELS - #############################
################################## - END - ##################################


################################ - START - #################################
############################## - EVALUATION - ##############################
MAE = make_scorer(mean_absolute_error)
# ACC = make_scorer(acc_fun)

# Run many iterations on each model
bestModelsList_regression = []
bestModelsList_classification = []
iterations = 100
modelsScoresDict_reg = {"Ridge": [],
                        "LogisticAT": [],
                        "LogisticRegression": [],
                        "MLPRegressor": [],
                        "DecisionTreeRegressor": [],
                        "SVR": []}
for i in range(0, iterations):
    # Split data in each iteration
    X, y, test_features, test_labels = splitData(updatedDataset)
    # print(i)
    scoresR = list()
    scoresC = list()
    # Train all models in each iteration
    for model in ml_models:
        # model.fit(X, y)

        # Regression
        score = cross_val_score(
            model, X, y, cv=5, scoring=MAE)
        score = score.mean()
        modelsScoresDict_reg[str(model).split("(")[0]].append(score)

        # scoresR.append(score)
        # print(score)

        # # Classification
        # score = cross_val_score(
        #     model, X, y, cv=5, scoring=ACC)
        # score = score.mean()
        # scoresC.append(score)
        # print(score)
    # Append the best model's name and score in a list of tuples
    # bestModelsList_regression.append(
    #     (ml_models[scoresR.index(min(scoresR))], min(scoresR)))
    # bestModelsList_classification.append(
    #     (ml_models[scoresC.index(max(scoresC))], max(scoresC)))

# # Extract best models names and scores
# bestModelNamesR = [i[0] for i in bestModelsList_regression]
# bestModelScoresR = [i[1] for i in bestModelsList_regression]
# # # Extract best models names and scores
# # bestModelNamesC = [i[0] for i in bestModelsList_classification]
# # bestModelScoresC = [i[1] for i in bestModelsList_classification]
# print("Best Model:")
# bestModelR_ = mode(bestModelNamesR)
# print(bestModelR_)
# bestModelScore = bestModelScoresR[bestModelNamesR.index(bestModelR_)]
# print(bestModelScore)

# print("Coefficients Up to Now:")
# bestModelR_ = LogisticAT(alpha=0.00)
# X, y, test_features, test_labels = splitData(updatedDataset)
# coefs = bestModelR_.fit(X, y).coef_
# norm_coefs = [round(x/sum(coefs), 2) for x in coefs]
# # print(coefs)
# tuples__ = list(zip(list(updatedDataset.iloc[:, :-1].columns), norm_coefs))
# print(sorted(tuples__, key=lambda x: x[1], reverse=True))


for x in modelsScoresDict_reg.keys():
    print(x, mean(modelsScoresDict_reg[x]))
# print(updatedDataset)

# # Collect the most frequent best model from the best models' list
# # creates a cloned estimator - removes fitted data
# print("Best Model:")
# bestModelR = clone(bestModelR_)
# bestModelScore = bestModelScoresR[bestModelNamesR.index(bestModelR_)]
# print(bestModelR)
# # print("MAE Score after feature adding and before feature selection:", end=" ")
# # print(bestModelScore)
# ############################## - EVALUATION - ##############################
# ################################# - END - ##################################


# ################################# - START - #################################
# ########################### - FEATURE SELECTION - ###########################
# df1 = updatedDataset[['minIntQoS', 'QoE']]
# df2 = updatedDataset[['minIntQoS', 'Int', 'QoE']]
# df3 = updatedDataset[['minIntQoS', 'Int', 'maxIntQoR', 'QoE']]
# df4 = updatedDataset[['minIntQoS', 'maxIntQoR', 'Int', 'QoS', 'QoE']]
# df5 = updatedDataset[['minIntQoS', 'maxIntQoR',
#                       'Int', 'QoS', 'minQoSQoR', 'QoE']]
# df6 = updatedDataset[['minIntQoS', 'maxIntQoR',
#                       'Int', 'QoS', 'minQoSQoR', 'maxQoSQoR', 'QoE']]
# df7 = updatedDataset[['minIntQoS', 'maxIntQoR', 'Int',
#                       'QoS', 'minQoSQoR', 'maxQoSQoR', 'QoR', 'QoE']]
# df8 = updatedDataset[['minIntQoS', 'maxIntQoR', 'Int',
#                       'QoS', 'minQoSQoR', 'maxQoSQoR', 'QoR', 'minIntQoR', 'QoE']]
# df9 = updatedDataset[['minIntQoS', 'maxIntQoR', 'Int',
#                       'QoS', 'minQoSQoR', 'maxQoSQoR', 'QoR', 'minIntQoR', 'maxIntQoS', 'QoE']]

# df1 = updatedDataset[['minIntQoS', 'QoE']]
# df2 = updatedDataset[['minIntQoS', 'Int', 'QoE']]
# df3 = updatedDataset[['minIntQoS', 'Int', 'QoS', 'QoE']]
# df4 = updatedDataset[['minIntQoS', 'Int', 'QoS', 'minIntQoR', 'QoE']]
# df5 = updatedDataset[['minIntQoS', 'Int',
#                       'QoS', 'minIntQoR', 'minQoSQoR', 'QoE']]
# df6 = updatedDataset[['minIntQoS', 'Int',
#                       'QoS', 'minIntQoR', 'minQoSQoR', 'maxQoSQoR', 'QoE']]
# df7 = updatedDataset[['minIntQoS', 'Int',
#                       'QoS', 'minIntQoR', 'minQoSQoR', 'maxQoSQoR', 'maxIntQoR', 'QoE']]
# df8 = updatedDataset[['minIntQoS', 'Int',
#                       'QoS', 'minIntQoR', 'minQoSQoR', 'maxQoSQoR', 'maxIntQoR', 'maxIntQoS', 'QoE']]
# df9 = updatedDataset[['minIntQoS', 'Int',
#                       'QoS', 'minIntQoR', 'minQoSQoR', 'maxQoSQoR', 'maxIntQoR', 'maxIntQoS', 'QoR', 'QoE']]
# dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
# modelsScoresDict = {"df1": [],
#                     "df2": [],
#                     "df3": [],
#                     "df4": [],
#                     "df5": [],
#                     "df6": [],
#                     "df7": [],
#                     "df8": [],
#                     "df9": []}

# model = LogisticAT(alpha=0.00)
# for i in range(0, 100):
#     for index, dff in enumerate(dfs):
#         X, y, test_features, test_labels = splitData(dff)
#         score = cross_val_score(model, X, y, cv=5, scoring=MAE)
#         score = score.mean()
#         modelsScoresDict["df" + str(index + 1)].append(score)
# means = []
# stds = []
# for x in modelsScoresDict.keys():
#     means.append(np.asarray(modelsScoresDict[x]).mean())
#     stds.append(np.asarray(modelsScoresDict[x]).std() * 1.96 / math.sqrt(len(X)))

# plt.ylabel("MAE of Ordinal Regression model")
# plt.xlabel("Number of Features used")
# # plt.bar(range(len(means)), height=means, fill=False)
# # plt.plot(means)
# plt.LineStyle = 'none'
# plt.Marker = 'o'
# plt.errorbar(x=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
#              y=means, yerr=stds, ls='-.', ecolor='red')
# plt.grid('on')
# plt.show()


# featuresList = []
# numofIters = 50
# for i in range(0, numofIters, 1):
#     X, y, testX, testY = splitData(updatedDataset)
#     bestModel = bestModelR.fit(X, y)
#     # Get important features' indexes
#     featuresIndexes = SelectFromModel(bestModel, prefit=True,
#                                       threshold='mean').get_support(indices=True)
#     features = [updatedDataset.columns[i] for i in featuresIndexes]
#     featuresList.append(features)

# # Frequency of Features
# flat_list = [item for sublist in featuresList for item in sublist]
# featFreq = dict(Counter(chain(flat_list)))
# freqDict = dict()
# for key in featFreq:
#     result = featFreq[key] / numofIters
#     freqDict[key] = result
# sorted_freqDict = sorted(freqDict.items(), reverse=True, key=lambda kv: kv[1])
# print("Frequency of features in " + str(numofIters) + " iterations:")
# pp.pprint(sorted_freqDict)
# threshold = 0.5
# featuresAboveThreshold = [x[0] for x in sorted_freqDict if x[1] > threshold]

# updatedDataset = updatedDataset[featuresAboveThreshold]
# updatedDataset['QoR'] = df_base['QoR']
# updatedDataset['QoS'] = df_base['QoS']
# updatedDataset['Int'] = df_base['Int']
# updatedDataset['QoE'] = qoeColumn

# # x, y = zip(*sorted_freqDict)  # unpack a list of pairs into two tuples
# # plt.plot(x, y)
# # plt.ylabel('some numbers')
# # plt.show()

# # Compare models - with/without added features, with/without feat. selection
# # creates a cloned estimator - removes fitted data
# bestModelR = clone(bestModelR_)
# _dfPrimary_ = df_base.iloc[:, :-1]
# X, y, test_features, test_labels = splitData(_dfPrimary_)
# score = cross_val_score(
#     bestModelR, X, y, cv=5, scoring=MAE)
# score = score.mean()
# print("MAE Score before feature adding:", end=" ")
# print(score)

# print("MAE Score after feature adding and before feature selection:", end=" ")
# print(bestModelScore)

# # creates a cloned estimator - removes fitted data
# bestModelR = clone(bestModelR_)
# minScore = 1
# for i in range(0, iterations):
#     X, y, test_features, test_labels = splitData(updatedDataset)
#     score = cross_val_score(
#         bestModelR, X, y, cv=5, scoring=MAE)
#     score = score.mean()
#     if score < minScore:
#         minScore = score
# print("MAE Score after feature adding and after feature selection:", end=" ")
# print(minScore)


# # Print correlation matrix among features
# print("Correlation of latent features:")
# print(updatedDataset.corr('pearson'))


# Print coefficients after feature selection
print("Coefficients after feature selection:")
bestModelR_ = LogisticAT(alpha=0.00)
bestModelR = clone(bestModelR_)
X, y, test_features, test_labels = splitData(updatedDataset, 0.2)
print(list(updatedDataset.iloc[:, :-1].columns))
coefs = bestModelR.fit(X, y).coef_
# print(coefs)
print([x/sum(coefs) for x in coefs])
print(mean_absolute_error(bestModelR.predict(test_features), test_labels))
# print(mean_absolute_error(LogisticRegression(multi_class='multinomial',
#                                              solver='lbfgs',
#                                              class_weight='balanced', max_iter=10000).predict(test_features), test_labels))
# print(mean_absolute_error(MLPRegressor(solver='lbfgs', alpha=1e-5,
#                                        hidden_layer_sizes=(16, 16), random_state=1).predict(test_features), test_labels))
# print(mean_absolute_error(Ridge(alpha=1.0).predict(test_features), test_labels))
# print(mean_absolute_error(SVR(gamma="auto").predict(test_features), test_labels))
X, y, test_features, test_labels = splitData(updatedDataset, 0.0)
coefs = bestModelR.fit(X, y).coef_
# print(coefs)
print([x/sum(coefs) for x in coefs])
########################### - FEATURE SELECTION - ##########################
################################# - END - ##################################
