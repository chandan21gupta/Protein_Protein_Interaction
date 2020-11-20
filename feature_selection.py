import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import RobustScaler
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

a = np.load('./positive.npy', allow_pickle=True)
b = np.load('./negative.npy', allow_pickle=True)
y_train = np.array([1 for i in range(len(a))] + [0 for i in range(len(b))])

print(len(a))
print(len(b))

X_train = np.vstack((a, b))

print(X_train.shape, y_train.shape)
# Let's say X_train is your input dataframe
from sklearn.preprocessing import MinMaxScaler

min_max_scalor = MinMaxScaler()
df = min_max_scalor.fit_transform(X_train)

# wrap it up if you need a dataframe
feature_columns = [i for i in range(100)]
df = pd.DataFrame(X_train, columns=feature_columns)
df['y'] = y_train

df.drop_duplicates(keep='first', inplace=True)

from sklearn.utils import shuffle

df = shuffle(df)
df.reset_index(inplace=True, drop=True)


# model = LogisticRegression()
#
# model.fit(df[feature_columns],y_train)
#
# importance = model.coef_[0]
#
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()

model = RandomForestClassifier()
# fit the model
importance = []
for i in range(30):
    model.fit(df[feature_columns], df['y'])
    # get importance
    importance.append(model.feature_importances_)
# summarize feature importance

importance = np.array(importance)
importance = np.mean(importance, axis=0)
a = []
index = 0
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
    a.append([index, v])
    index += 1
#
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()

a = [x for x in range(len(importance))]
b = []
for i in range(len(a)):
    b.append([i, importance[i]])

from heapq import nlargest

c = nlargest(3, b, key=lambda e: e[1])
print(c)
d = []
for i in c:
    d.append(i[0])

#Compute the correlation matrix
corr = df[d].corr()
print(corr)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#