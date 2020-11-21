import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def bar_plot(x, data):

	sns.countplot(x, data)
	plt.show()

def hist_plot(x, data):
	sns.set_theme(style="white")

	sns.displot(data, x = x)
	plt.show()

def generate_correlation_matrix(data):
	matrix = data.corr()
	mask = np.triu(np.ones_like(matrix, dtype=bool))
	f, ax = plt.subplots(figsize = (11,9))
	cmap = sns.diverging_palette(230, 20, as_cmap = True)

	sns.heatmap(matrix, mask = mask, cmap = cmap, center = 0, square = True, vmax = 0.7, vmin = -0.7)

	plt.show()


data = pd.read_csv('final_dataset.csv', index_col = [0])

# print(data[''])

# X = StandardScaler().fit_transform(data[data.columns[0:100]])
# # pca = PCA(2)
# # principal_components = pca.fit_transform(X)
# # pca = PCA().fit(X)
# # plt.plot(np.cumsum(pca.explained_variance_ratio_))
# # plt.xlabel('number of components')
# # plt.ylabel('cumulative explained variance')
# # print(pca.explained_variance_ratio_)
# # plt.show()
# pca = PCA(3)
# principalComponents = pca.fit_transform(X)
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
# plt.scatter(principalDf['principal component 1'],principalDf['principal component 2'],s=30,c='goldenrod',alpha=0.5)
# plt.title('plotting both variables')
# plt.xlabel('Principal component 1')
# plt.ylabel('Principal component 2')
# plt.show()





