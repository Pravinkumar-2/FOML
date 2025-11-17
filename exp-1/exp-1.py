#FOMLEXP01
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('/content/iriss.csv')
df = df.rename(columns={'sepal_length': 'sepal.length', 'sepal_width': 'sepal.width', 'petal_length': 'petal.length', 'petal_width': 'petal.width', 'species': 'species'})
df.head(150)
#univariate for sepal width
df_Setosa=df.loc[df['species']=='Iris-setosa']
df_Virginica=df.loc[df['species']=='Iris-virginica']
df_Versicolor=df.loc[df['species']=='Iris-versicolor']
plt.scatter(df_Setosa['sepal.width'],np.zeros_like(df_Setosa['sepal.width']))
plt.scatter(df_Virginica['sepal.width'],np.zeros_like(df_Virginica['sepal.width']))
plt.scatter(df_Versicolor['sepal.width'],np.zeros_like(df_Versicolor['sepal.width']))
plt.xlabel('sepal.width')
plt.show()

#univariate for sepal length
df_Setosa=df.loc[df['species']=='Iris-setosa']
df_Virginica=df.loc[df['species']=='Iris-virginica']
df_Versicolor=df.loc[df['species']=='Iris-versicolor']
plt.scatter(df_Setosa['sepal.length'],np.zeros_like(df_Setosa['sepal.length']))
plt.scatter(df_Virginica['sepal.length'],np.zeros_like(df_Virginica['sepal.length']))
plt.scatter(df_Versicolor['sepal.length'],np.zeros_like(df_Versicolor['sepal.length']))
plt.xlabel('sepal.length')
plt.show()
#univariate for petal width
df_Setosa=df.loc[df['species']=='Iris-setosa']
df_Virginica=df.loc[df['species']=='Iris-virginica']
df_Versicolor=df.loc[df['species']=='Iris-versicolor']
plt.scatter(df_Setosa['petal.width'],np.zeros_like(df_Setosa['petal.width']))
plt.scatter(df_Virginica['petal.width'],np.zeros_like(df_Virginica['petal.width']))
plt.scatter(df_Versicolor['petal.width'],np.zeros_like(df_Versicolor['petal.width']))
plt.xlabel('petal.width')
plt.show()

#univariate for petal length
df_Setosa=df.loc[df['species']=='Iris-setosa']
df_Virginica=df.loc[df['species']=='Iris-virginica']
df_Versicolor=df.loc[df['species']=='Iris-versicolor']
plt.scatter(df_Setosa['petal.length'],np.zeros_like(df_Setosa['petal.length']))
plt.scatter(df_Virginica['petal.length'],np.zeros_like(df_Virginica['petal.length']))
plt.scatter(df_Versicolor['petal.length'],np.zeros_like(df_Versicolor['petal.length']))
plt.xlabel('petal.length')
plt.show()
#bivariate sepal.width vs petal.width
sns.FacetGrid(df,hue='species',height=5).map(plt.scatter,"sepal.width","petal.width").add_legend();

plt.show()
#bivariate sepal.length vs petal.length
sns.FacetGrid(df,hue='species',height=5).map(plt.scatter,"sepal.length","petal.length").add_legend();
plt.show()

#multivariate all the features
sns.pairplot(df,hue="species",height=2)
