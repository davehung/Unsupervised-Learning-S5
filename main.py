from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#dave's assignment 2

images, target = fetch_olivetti_faces(return_X_y=True)

sss = StratifiedShuffleSplit(test_size=.2, random_state=0)
for data_index, test_index in sss.split(images, target):
     x, x_test = images[data_index], images[test_index]
     y, y_test = target[data_index], target[test_index]



sss1 = StratifiedShuffleSplit(test_size=.25, random_state=0)

for train_index, val_index in sss1.split(x, y):
     x_train, x_val = x[train_index], x[val_index]
     y_train, y_val = y[train_index], y[val_index]


LR = LogisticRegression(random_state=0)
scores = cross_val_score(LR, images, target)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())



plt.subplot(1, 2, 1)
plt.hist(y_val)
plt.title('training  distribution')
plt.xlabel('class')
plt.ylabel('number')
plt.subplot(1, 2, 2)
plt.hist(y_test)
plt.title('test distribution')
plt.xlabel('class')
plt.ylabel('number')
plt.show()

k=[]
silScores=[]
bestScore=0
for i in range(20, 35):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(images)
    silScore = silhouette_score(images, kmeans.labels_)
    if silScore > bestScore:
        SilTop = i
        bestScore=silScore
    k.append(i)
    print('for n_clusters =', i, 'The average silhouette_score is:', silScore)
    silScores.append(silScore)


p=plt.plot(k,silScores)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")



n_clusters=SilTop
img_reduced = KMeans(n_clusters=n_clusters).fit_predict(images)




LR = LogisticRegression(random_state=0)
scores = cross_val_score(LR, images, img_reduced)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())
