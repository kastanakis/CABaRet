import json
import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, Ridge
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_regression, mutual_info_regression, RFECV
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.utils import resample
from sklearn.utils.validation import check_X_y
from statistics import mode, mean
from sklearn import base, metrics
from sklearn.base import clone
from mord import LogisticAT, OrdinalRidge, LAD
from functools import reduce
from itertools import chain
from collections import Counter
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
FONTSIZE = 20


pp = pprint.PrettyPrinter(indent=4)  # Pretty Printers
pd.options.mode.chained_assignment = None  # Silence warnings

# Separate features from labels from training/test set


def splitData(df):
    # Split to 80 20 -- use CV for train split
    train, test = train_test_split(df, test_size=0.2, shuffle=True)
    X = train.iloc[:, :-1]
    y = train.iloc[:, -1]
    test_features = test.iloc[:, :-1]
    test_labels = test.iloc[:, -1]
    return X, y, test_features, test_labels


################################# - START - #################################
############################# - PREPROCESSING - #############################
# Read the json file
with open("CABaRet/MLanalysis/ratings_1-0.json") as f:
    data = json.load(f)
    
# Extract current columns
df_base = pd.DataFrame(data, columns=["QoS", "QoR", "Int", "QoE", 'Hit'])

# # Remove Cache Hit - not necessary for QoE = f(QoR, Int, QoS)
df = df_base.iloc[:, :-1]
# df = df_base
print(df)
ind = np.where((np.maximum(df['QoS'].values, df['Int'].values) >= df["QoE"].values) &
               (np.minimum(df['QoS'].values, df['Int'].values) <= df["QoE"].values))
df = df.loc[ind]
print(df)
############################# - PREPROCESSING - #############################
################################## - END - ##################################


################################ - START - #################################
############################# - CRAFT MODELS - #############################
# Models to Train
ml_models = [\
    # LogisticRegression(multi_class='ovr',
    #                    solver='liblinear',
    #                    class_weight='balanced'),\
    LogisticRegression(multi_class='multinomial',
                       solver='lbfgs',
                       class_weight='balanced', max_iter=10000),\
    # RandomForestClassifier(class_weight='balanced', n_estimators=100,
    #                        oob_score=True, random_state=123),\
    DecisionTreeRegressor(),
    # SVC(kernel='rbf', gamma="auto"),\
    # SVC(kernel='poly', gamma="auto"),\
    # SVC(kernel='sigmoid', gamma="auto"),\
    SVR(gamma="auto"),\
    # MLPRegressor(solver='lbfgs', alpha=1e-5,
    #               hidden_layer_sizes=(4, 4), random_state=1),\
    # MLPRegressor(solver='lbfgs', alpha=1e-5,
    #               hidden_layer_sizes=(8, 8, 8), random_state=1),\
    MLPRegressor(solver='lbfgs', alpha=1e-5,
                 hidden_layer_sizes=(16, 16), random_state=1),\
    Ridge(alpha=1.0),\
    LogisticAT(alpha=0.00)
]


# Select the ones you want
# df1 = df[['QoR', 'QoS', 'QoE']]
df2 = df[['Int', 'QoS', 'QoE']]
# df3 = df[['Int', 'QoR', 'QoE']]


############################## - CRAFT MODELS - #############################
################################## - END - ##################################


################################ - START - #################################
############################## - EVALUATION - ##############################
MAE = make_scorer(mean_absolute_error)
modelsScoresDict = {
    "LogisticRegression": [],
    # "RandomForestClassifier": [],
    "DecisionTreeRegressor": [],
    # "SVC": [],
    "SVR": [],
    "MLPRegressor": [],
    "Ridge": [],
    "LogisticAT": []}
# Train all models in each iteration
for i in range(0, 100):
    # Split data in each iteration
    # X, y, test_features, test_labels = splitData(df1)
    X, y, test_features, test_labels = splitData(df2)
    # X, y, test_features, test_labels = splitData(df3)
    for model in ml_models:
        # model.fit(X, y)

        # Regression
        score = cross_val_score(
            model, X, y, cv=5, scoring=MAE)
        score = score.mean()
        modelsScoresDict[str(model).split("(")[0]].append(score)


for x in modelsScoresDict.keys():
    print(x, mean(modelsScoresDict[x]))
############################# - EVALUATION - ##############################
################################ - END - ##################################


def plot_class_boundaries_2D(model, X, y, feature_names=None):
    # create a mesh to plot in
    x_min, x_max = 0, 6
    y_min, y_max = 0, 6
    h = 0.05  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundaries and classes
    Z = np.round(model.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plot = plt.contour(xx, yy, Z, colors='black',
                       linestyles='dashed', linewidths=2)
    plt.pcolormesh(xx, yy, Z, cmap='winter',
                   norm=colors.Normalize(-1, 6), zorder=0)

    # format axis, etc.
    plt.xticks(range(1, 6))
    plt.yticks(range(1, 6))
    if feature_names is not None:
        plt.xlabel(feature_names[0], fontsize=FONTSIZE)
        plt.ylabel(feature_names[1], fontsize=FONTSIZE)
    e = 0.3
    plt.axis([1-e, 5+e, 1-e, 5+e])
    plt.grid(True)
    plt.show()

    return plot


# REMOVE_OUTLIERS = True
# if REMOVE_OUTLIERS:
#     ind = np.where((np.maximum(df['QoS'].values, df['Int'].values, df['QoR'].values) >= df["QoE"].values) &
#                    (np.minimum(df['QoS'].values, df['Int'].values, df['QoR'].values) <= df["QoE"].values))

df = df.loc[ind]

df1 = df[['QoR', 'QoS', 'QoE']]
X1, y1, test_features1, test_labels1 = splitData(df1)
bestModelFitted_df1_mlp = MLPRegressor(solver='lbfgs', alpha=1e-5,
                                       hidden_layer_sizes=(16, 16), random_state=1).fit(X1, y1)
bestModelFitted_df1_ord = LogisticAT(alpha=0.00).fit(X1, y1)
# plot = plot_class_boundaries_2D(
#     bestModelFitted_df1_mlp, X1, y1, feature_names=['QoR', 'QoS'])
# plot = plot_class_boundaries_2D(
#     bestModelFitted_df1_ord, X1, y1, feature_names=['QoR', 'QoS'])

df2 = df[['Int', 'QoS', 'QoE']]
REMOVE_OUTLIERS = True
if REMOVE_OUTLIERS:
    ind = np.where((np.maximum(df['QoS'].values, df['Int'].values) >= df["QoE"].values) &
                   (np.minimum(df['QoS'].values, df['Int'].values) <= df["QoE"].values))
print(df2)

X2, y2, test_features2, test_labels2 = splitData(df2)
bestModelFitted_df2_mlp = MLPRegressor(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(16, 16), random_state=1).fit(X2, y2)
# bestModelFitted_df2_ord = LogisticAT(alpha=0.00).fit(X2, y2)

plot = plot_class_boundaries_2D(
    bestModelFitted_df2_mlp, X2, y2, feature_names=['Interest Ratings', 'QoS Ratings'])
# plot = plot_class_boundaries_2D(
#     bestModelFitted_df2_ord, X2, y2, feature_names=['Int', 'QoS'])
