import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pandas_profiling import ProfileReport
a = np.load('./positive.npy',allow_pickle=True)
b = np.load('./negative.npy',allow_pickle=True)
y_train = np.array([1 for i in range(len(a))] + [0 for i in range(len(b))])

print(len(a))
print(len(b))


X_train = np.vstack((a,b))

print(X_train.shape,y_train.shape)
# Let's say X_train is your input dataframe
from sklearn.preprocessing import MinMaxScaler
minmaxscalor = MinMaxScaler()
X_train_norm = minmaxscalor.fit_transform(X_train)




# wrap it up if you need a dataframe
feature_columns = [i for i in range(100)]
df = pd.DataFrame(X_train_norm,columns=feature_columns)
df['y'] = y_train


 #create an array with the index of the rows where missing data are present

print('Size of the dataframe: {}'.format(df.shape))

df.drop_duplicates(keep='first',inplace=True)

print('After Dropping Duplicates')
print(df.shape)







#
# prof = ProfileReport(df,minimal=True)
# prof.to_file('output_standardized.html')
# # np.random.seed(42)
# # rndperm = np.random.permutation(df.shape[0])
#
# # prof = ProfileReport(df,minimal=True)
# # prof.to_file(output_file='Report_2.html')
# df = df.astype(float)
# f, ax = plt.subplots(figsize=(11, 9))
# sns.heatmap(df)
# #
# # # Compute the correlation matrix
# # corr = df.corr()
# # print(corr)
# # # Generate a mask for the upper triangle
# # mask = np.triu(np.ones_like(corr, dtype=bool))
# #
# # # Set up the matplotlib figure
# # f, ax = plt.subplots(figsize=(11, 9))
# #
# # # Generate a custom diverging colormap
# # cmap = sns.diverging_palette(230, 20, as_cmap=True)
# #
# # # Draw the heatmap with the mask and correct aspect ratio
# # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
# #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# #
# #

pca = PCA(n_components=50)
pca_result = pca.fit_transform(df[feature_columns].values)
# print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
print(pca_result)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]

plt.figure()
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

plt.show()







tsne = TSNE(n_components=2, verbose=0, perplexity=15, n_iter=250)
tsne_pca_results = tsne.fit_transform(df[feature_columns])



df['tsne-pca50-one'] = tsne_pca_results[:,0]
df['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure()





sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.3
)

plt.show()