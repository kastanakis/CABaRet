import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# import ML models
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, LogisticRegression
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, accuracy_score


INPUT_FILE = 'CABaRet\\MLanalysis\\ratings_1-0.txt'
REMOVE_OUTLIERS = True
PERCENTAGE_TRAIN = 0.99#8
FONTSIZE = 20


DICT_OF_MODELS = {\
	'Ridge': Ridge(),\
	'Logistic': LogisticRegression(),\
	'LDA': LinearDiscriminantAnalysis(),\
	# 'QDA': QuadraticDiscriminantAnalysis(),\
	'SVR': SVR(),\
	'SVC': SVC(),\
	# 'GaussianNB': GaussianNB(),\
	'MLPRegressor': MLPRegressor((16,16,))
	}

def set_ratings(data):
	QoS = data[:,0]
	QoR = data[:,1]
	Int = data[:,2]
	QoE = data[:,3]
	Hit = data[:,4]
	return QoS,QoR,Int,QoE,Hit



def plot_class_boundaries_2D(model, X, Y, plot_scatter=True, plot_class_means=True, feature_names=None):
	# create a mesh to plot in
	x_min, x_max = 0, 6
	y_min, y_max = 0, 6
	h = 0.05  # step size in the mesh
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


	# Plot the decision boundaries and classes
	Z = np.round(model.predict(np.c_[xx.ravel(), yy.ravel()]))
	Z = Z.reshape(xx.shape)
	plot = plt.contour(xx, yy, Z, colors='black', linestyles='dashed', linewidths=2)
	plt.pcolormesh(xx, yy, Z, cmap='winter', norm=colors.Normalize(-1, 6), zorder=0)

	# plot the means of clusters
	if plot_class_means:
		try:
			class_centers = model.means_
			for i in range(5):
				plt.plot(class_centers[i,0], class_centers[i,1], marker='o', color='yellow', markersize=15, markeredgecolor='black')
				plt.plot(class_centers[i,0], class_centers[i,1], marker='${}$'.format(i+1), color='black', markersize=10, markeredgecolor='black')
		except:	
			pass

	# plot the classification results
	if plot_scatter:
		add_noise = lambda x : 0.2-0.4*np.random.random(x)
		Y_result = np.round(model.predict(X))
		IND = np.where(Y == Y_result)
		shape = np.shape(X[IND][:,0])
		plt.plot(X[IND][:,0]+add_noise(shape), X[IND][:,1]+add_noise(shape),'.',  color='black', marker='o')
		IND = np.where(abs(Y - Y_result)==1)
		shape = np.shape(X[IND][:,0])
		plt.plot(X[IND][:,0]+add_noise(shape), X[IND][:,1]+add_noise(shape),'.', color='white', marker='+', markersize=7)
		IND = np.where(abs(Y - Y_result)>=2)
		shape = np.shape(X[IND][:,0])
		plt.plot(X[IND][:,0]+add_noise(shape), X[IND][:,1]+add_noise(shape),'.', color='red', marker='x', markersize=7)

	# format axis, etc.
	plt.xticks(range(1,6))
	plt.yticks(range(1,6))
	if feature_names is not None:
		plt.xlabel(feature_names[0], fontsize=FONTSIZE)
		plt.ylabel(feature_names[1], fontsize=FONTSIZE)
	e = 0.3
	plt.axis([1-e,5+e,1-e,5+e])
	plt.grid(True)

	return plot



# load data
data = np.genfromtxt(INPUT_FILE, delimiter=',', dtype=int, skip_header=1)
QoS,QoR,Int,QoE,Hit = set_ratings(data)

# data preprocessing to remove "outliers"
if REMOVE_OUTLIERS:
	ind = np.where((np.maximum(QoS,Int,QoR)>=QoE) & (np.minimum(QoS,Int,QoR)<=QoE))
	QoS,QoR,Int,QoE,Hit = set_ratings(data[ind])

# data spliting into training/testing sets
data_size = len(QoE)
train_indices = np.random.choice(range(data_size), round(PERCENTAGE_TRAIN*data_size), replace=False)
test_indices = list(set(range(data_size)) - set(train_indices))


# FEATURE_COMBINATIONS = [\
# 						[QoE, QoS, Int],\
# 						[QoE, QoS, QoR],\
# 						[QoE, QoR, Int],\
# 						[QoE, np.minimum(QoS,Int), np.maximum(QoS,Int)]\
# 						]


FEATURE_COMBINATIONS = {\
						'QoS,Int': [QoS, Int],\
						'QoS,QoR': [QoS, QoR],\
						'QoR,Int': [QoR, Int],\
						'min_QoS_Int,max_QoS_Int': [np.minimum(QoS,Int), np.maximum(QoS,Int)]\
						}


for feature_names, features in FEATURE_COMBINATIONS.items():
	print('-----------------------------')
	print('features: '+feature_names)
	print('-----------------------------')

	F = np.asarray(features,order='F').T

	X_train = F[train_indices,:]
	Y_train = QoE[train_indices]
	X_test = F[test_indices,:]
	Y_test = QoE[test_indices]

	for model_name, model in DICT_OF_MODELS.items():
		model.fit(X_train, Y_train)

		# print performance
		predictions = model.predict(X_test)
		predictions_rounded = np.round(predictions)
		predictions_train = model.predict(X_train)
		predictions_train_rounded = np.round(predictions_train)
		
		print(model_name)
		# print("\t test MAE (regression) : {} ({})".format(round(mean_absolute_error(Y_test, predictions),2), round(median_absolute_error(Y_test, predictions),2)))
		# print("\t test MAE (rounded)    : {} ({})".format(round(mean_absolute_error(Y_test, predictions_rounded),2), round(median_absolute_error(Y_test, predictions_rounded),2)))
		print("\t train MAE (regression): {} ({})".format(round(mean_absolute_error(Y_train, predictions_train),2), round(median_absolute_error(Y_train, predictions_train),2)))
		print("\t train MAE (rounded)   :{} ({})".format(round(mean_absolute_error(Y_train, predictions_train_rounded),2), round(median_absolute_error(Y_train, predictions_train_rounded),2)))
		print("\t train Accuracy        :{}".format(round(accuracy_score(Y_train, predictions_train_rounded),2)))
	

		# plot 2D 
		feature1_name = feature_names.split(',')[0]
		feature2_name = feature_names.split(',')[1]
		plot = plot_class_boundaries_2D(model, X_train, Y_train, plot_scatter=False, plot_class_means=False, feature_names=[feature1_name,feature2_name])
		filename = 'fig_class_boundaries_2D__{}_vs_{}__{}__rm_outliers_{}.eps'.format(feature1_name, feature2_name, model_name, REMOVE_OUTLIERS)
		# plt.savefig(filename)
		# plt.close()
		
		print(' ')
	print(' ')
