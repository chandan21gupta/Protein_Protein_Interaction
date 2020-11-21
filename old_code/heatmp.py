import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pandas_profiling import ProfileReport



df = pd.read_pickle('standardized.pkl')

df2 = pd.read_csv('raw.csv')
x = df2.to_numpy()
x = x.T
y = x[-1].T
x = x[1:-1].T

y = y.reshape(-1,1)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5,shuffle=True)
clf = RandomForestClassifier(random_state=0)
f1 = []
accuracy = []
for train_index, test_index in skf.split(x, y):
     X_train, X_test = x[train_index], x[test_index]
     y_train, y_test = y[train_index], y[test_index]
     clf.fit(X_train, y_train)
     y_pred = clf.predict(X_test)
     # score = metrics.f1_score(y_test, y_pred, pos_label=list(set(y_test)))
     pscore = metrics.accuracy_score(y_test, y_pred)
     # f1.append(score)
     accuracy.append(pscore)

print(accuracy)
accuracy = np.array(accuracy)

print(np.average(accuracy))

from sklearn.datasets import make_classification



# x = x.T
# print(x)
# print(x.shape)
# for i in range(100):
#     plt.boxplot(x[i],labels=[i])
#     plt.show()

# i = [28,27,21,20,15,5,4,0]
# for j in i:
#     plt.hist(x[j],label = str(j))
#     plt.show()
#




# features = []
# for i in range(100):
#     features.append(i)
#
# pca = PCA(n_components=2)
# pc = pca.fit_transform(df[features])

# x = list(df['y'])
# print(x)
# for i in range(len(pc)):
#     if x[i] == 1:
#         plt.scatter(pc[i][0],pc[i][1],c='r')
#
#     elif x[i] == 0:
#         plt.scatter(pc[i][0],pc[i][1],c='b')
# plt.show()






# boxplot = df.boxplot(column=[28,27,21,20,15,5,4,0])
#
# plt.show()

# ax = sns.countplot(x="class", hue="who", data= df)