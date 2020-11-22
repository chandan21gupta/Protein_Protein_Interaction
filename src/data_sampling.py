'''
This function generates the training and testing dataset to be used for training and testing purposes with various models respectively. 
'''
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def generate_train_test_data(split = 0.3, random_state = 42, scaling = 'standard', sampling = True, show_plots = True):

	'''
	This method generates the training and test data. 

	Parameters :- split : Float, (0,1) - fraction of size of testing data to be used to test the model.
				  random_state : Integer - state of the randomization of the sampling data.
				  scaling : String, 'normal_negative', 'normal_positive', 'standard' - Type of scaling to be used, scaling data by 
				  																		[-1,1], [0,1], Standard Normal respectively
				  sampling : Boolean - To generate synthetic data using training data with SMOTE
				  show_plots : Boolean - To show target variable distribution after each sampling method.											
	'''

	data = pd.read_csv('../extracted_datasets/raw.csv', index_col = [0])

	X = data[data.columns[0:100]]

	y = data['y']
    
	if show_plots == True:

		sns.countplot(y).set_title('Target Distribution in whole dataset')
		plt.show()

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state, stratify = y, test_size = split)


	if scaling == 'normal_negative':
		scale = MinMaxScaler(feature_range = (-1,1))

	elif scaling == 'normal_positive':
		scale = MinMaxScaler()

	elif scaling == 'standard':
		scale = StandardScaler()

	if sampling == False:

		X_train = scale.fit_transform(X_train)
		X_test = scale.transform(X_test)

		if show_plots == True:
			sns.countplot(x = y_train).set_title('Target Distribution in Training Dataset')
			plt.show()
			sns.countplot(x = y_test).set_title('Target Distribution in Test dataset')
			plt.show()

		return X_train, X_test, y_train, y_test

	else:

		if show_plots == True:
			sns.countplot(x = y_train).set_title('Target Distribution in Training Dataset without SMOTE')
			plt.show()

		oversample = SMOTE()
		X_train, y_train = oversample.fit_resample(X_train, y_train)

		X_train = scale.fit_transform(X_train)
		X_test = scale.transform(X_test)

		if show_plots == True:
			sns.countplot(x = y_train).set_title('Target Distribution in Training Dataset with SMOTE')
			plt.show()
			sns.countplot(x = y_test).set_title('Target Distribution in Test dataset')
			plt.show()
		return X_train, X_test, y_train, y_test

if __name__ == '__main__':
	generate_train_test_data()

