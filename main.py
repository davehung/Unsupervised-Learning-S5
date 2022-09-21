import pandas as pd,numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA
print('Question 1:')
df_dave=fetch_openml('mnist_784')
print(df_dave.keys())
x,y=df_dave['data'],df_dave['target']
pca=PCA(n_components=2)
x2d=pca.fit_transform(x)
print('EV ratio:')
print(pca.explained_variance_ratio_)

points=[0 for _ in range(x2d.shape[0])]
plt.title('Projection of first component')
plt.scatter(x2d[:,0],points)
plt.show()
plt.title('Projection of second component')
plt.scatter(x2d[:,1],points)
plt.show()
pca=IncrementalPCA(n_components=154)
for batch in np.array_split(x,100):
  pca.partial_fit(batch)
x_reduced=pca.transform(x)
x_original=pd.DataFrame(pca.inverse_transform(x_reduced))

#def plotter (r):
  #origin,compress=r[:784],r[784:]
  #digit=origin.values.reshape(28,28)
  #plt.axis('off')
  #plt.imshow(digit)
  #plt.show()
  #digit=compress.values.reshape(28,28)
  #plt.axis('off')
  #plt.imshow(digit)
  #plt.show()
#digits=pd.concat([x,x_recovered],axis=1)
#digits.apply(plotter,axis=1)
print('Question 2')
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
x,t=make_swiss_roll(n_samples=1000)
plot=plt.figure().add_subplot(projection='3d')
plot=plt.scatter(x[:,0],x[:,1],x[:,2],c=t)
plt.show()
pca=KernelPCA(n_components=2)
x2d=pca.fit_transform(x)
plt.scatter(x2d[:,0],x2d[:,1],c=t)
plt.show()

pca=KernelPCA(n_components=2,kernel='rbf')
x2d=pca.fit_transform(x)
plt.scatter(x2d[:,0],x2d[:,1],c=t)
plt.show()

pca=KernelPCA(n_components=2,kernel='sigmoid')
x2d=pca.fit_transform(x)
plt.scatter(x2d[:,0],x2d[:,1],c=t)
plt.show()
t=np.where(t<=t.mean(),0,1)
pipeline=Pipeline([('pca',KernelPCA(n_components=2)),('model',LogisticRegression())])
param={'pca__kernel': ['linear', 'rbf', 'sigmoid'], 'pca__gamma': np.arange(0, 1, .02), 'model__max_iter': [1000, 2000]}
grid_search=GridSearchCV(pipeline,param,scoring='accuracy',refit=True)
gs=grid_search.fit(x,t)
print('best estimator:')
print(gs.best_estimator_)
print('best score:')
print(gs.best_score_)



best = KernelPCA(n_components=2, kernel="rbf", gamma=0.08)
best_reduced = best.fit_transform(x)

plt.scatter(best_reduced[:, 0], best_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.show()