#FOMLEXP06
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)

# read data
recipes = pd.read_csv('/content/recipes_muffins_cupcakes.csv')
print(recipes.head())
print(recipes.shape)

# plot data
sns.lmplot(x='Sugar', y='Flour', data=recipes, hue='Type', palette='Set1', fit_reg=False,
           scatter_kws={"s":70})
plt.show()

# prepare data
X = recipes[['Sugar','Flour']].values
y = np.where(recipes['Type']=='Muffin', 0, 1)

# train model
model = svm.SVC(kernel='linear')
model.fit(X, y)

# get decision boundary
w = model.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(X[:,0].min(), X[:,0].max(), 100)
yy = a*xx - (model.intercept_[0]/w[1])

# margin lines via support vectors
b0 = model.support_vectors_[0]
yy_down = a*xx + (b0[1] - a*b0[0])
b1 = model.support_vectors_[-1]
yy_up = a*xx + (b1[1] - a*b1[0])

# re-plot with decision boundary and margin
sns.lmplot(x='Sugar', y='Flour', data=recipes, hue='Type', palette='Set1', fit_reg=False,
           scatter_kws={"s":70})
plt.plot(xx, yy, 'k-', linewidth=2)
plt.plot(xx, yy_down, 'k--', linewidth=2)
plt.plot(xx, yy_up, 'k--', linewidth=2)
plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1],
            s=80, facecolors='none', edgecolors='k')
plt.show()

# train/test split and evaluate
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model1 = svm.SVC(kernel='linear')
model1.fit(X_train, y_train)
pred = model1.predict(X_test)

print(pred)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
