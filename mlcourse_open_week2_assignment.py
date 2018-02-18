import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

sns.set()

# importing the data frame
df = pd.read_csv()
"F:/Deep Learning/mlcourse_open/telecom_churn.csv"
features = ['Total day minutes','Total intl calls']
df[features].hist(figsize=(12,4))
df[features].plot(kind='density', subplots=True, layout=(1, 2), sharex=False, figsize=(12,4))

# Seaborn distribution plot
sns.distplot(df['Total intl calls'])

# Seaborn box plot
sns.boxplot(data=df['Total intl calls'])

# Describe function
df[features].describe()

# Value counts
df['Churn'].value_counts()

# Count plot
sns.countplot(x='Churn',data=df)
sns.countplot(x='Customer service calls', data=df)

# Correlation matrix
numerical = list(set(df.columns) - set(['State', 'International plan', 'Voice mail plan', 'Area code', 'Churn',
                                      'Customer service calls']))

corr_matrix = df[numerical].corr()
sns.heatmap(corr_matrix)

# Refining the numerical list
numerical = list(set(numerical) -
                 set(['Total day charge', 'Total eve charge',
                      'Total night charge', 'Total intl charge']))

# Scatter plot
sns.jointplot(x='Total day minutes', y='Total night minutes', data=df, kind='scatter')
sns.jointplot(x='Total day minutes', y='Total night minutes', data=df, kind='kde')
sns.pairplot(df[numerical])

sns.lmplot('Total day minutes', 'Total night minutes', data=df, hue='Churn', fit_reg=False)

numerical.append('Customer service calls')
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 7))
for idx, feat in enumerate(numerical):
    ax = axes[int(idx / 4), idx % 4]
    sns.boxplot(x='Churn', y=feat, data=df, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(feat)
fig.tight_layout()

sns.boxplot(x='Churn', y='Total day minutes', data=df)
sns.violinplot(x='Churn', y='Total day minutes', data=df)

sns.factorplot(x='Churn', y='Total day minutes',
               col='Customer service calls',
               data=df[df['Customer service calls'] < 8],
               kind="box", col_wrap=4, size=3, aspect=.8)

sns.countplot(x='Customer service calls', hue='Churn', data=df)

sns.countplot(x='International plan', hue='Churn', data=df)
sns.countplot(x='Voice mail plan', hue='Churn', data=df)

pd.crosstab(df['State'], df['Churn']).T
df.groupby(['State'])['Churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X = df.drop(['Churn', 'State'], axis=1)
X['International plan'] = X['International plan'].map({'Yes': 1, 'No': 0})
X['Voice mail plan'] = X['Voice mail plan'].map({'Yes': 1, 'No': 0})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(random_state=17)
tsne_repr = tsne.fit_transform(X_scaled)
plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1])

# Assignment Start
Data = pd.read_csv("F:/Deep Learning/mlcourse_open/mlbootcamp5_train.csv", sep=';')
Data.shape

# 1. How many men and women are present in this dataset? Values of the gender feature were not expailned (whether "1" stands for women or men) – figure this out by looking analyzing height, reasonably assuming that on average men are taller.
Data['gender'].value_counts()
Data.groupby(['gender'])['height'].agg(np.mean)

# 2. Who more often report consuming alcohol - men or women?
Data.groupby(['gender'])['alco'].agg(np.mean)

# 3. What's the rounded difference between the percentages of smokers among men and women?
Data.groupby(['gender'])['smoke'].agg(np.mean)

# 4. What's the rounded difference between median values of age for smokers and non-smokers? You'll need to figure out the units of feature age in this dataset.
Data.groupby(['smoke'])['age'].agg(np.median)

# 5. Calculate fractions of ill people (with CVD) in two segments described in the task. What's the quotient of these two fractions?
Data['age_years'] = Data['age']/365
X = Data[(Data['age_years'] >= 60) & (Data['age_years'] <= 64) & (Data['gender'] == 2)]
X.shape
X1 = X[(X['ap_hi'] >= 160) & (X['ap_hi'] < 180) & (X['cholesterol'] == 3)]
X1['smoke'].value_counts()

X2 = X[(X['ap_hi'] >= 0) & (X['ap_hi'] < 120) & (X['cholesterol'] == 1)]
X2['smoke'].value_counts()

# 1.6. Choose the correct statements
Data['height_mts'] = Data['height'] / 100
Data['BMI'] = Data['weight'] / (Data['height_mts']*Data['height_mts'])
np.median(Data['BMI'])
Data.groupby(['gender'])['BMI'].agg(np.mean)
Data.groupby(['cardio'])['BMI'].agg(np.mean)

# 1.7. How many percents of data (rounded) did we throw away?
D = Data[Data['ap_lo'] > Data['ap_hi']]
H1 = Data[Data['height'] < Data['height'].quantile(q=0.025)]
H2 = Data[Data['height'] > Data['height'].quantile(q=0.975)]
W1 = Data[Data['weight'] < Data['weight'].quantile(q=0.025)]
W2 = Data[Data['weight'] > Data['weight'].quantile(q=0.975)]

# 2.1. Which pair of features has the strongest Pearson's correlation with gender feature?
Data['cardio'].corr(Data['cholesterol'])
Data['height'].corr(Data['smoke'])
Data['smoke'].corr(Data['alco'])
Data['height'].corr(Data['weight'])

# 2.2. Which pair of features has the strongest Spearman's correlation between each other?
Data['height'].corr(Data['weight'], method='spearman')
Data['age'].corr(Data['weight'], method='spearman')
Data['cholesterol'].corr(Data['gluc'], method='spearman')
Data['cardio'].corr(Data['cholesterol'], method='spearman')
Data['ap_hi'].corr(Data['ap_lo'], method='spearman')
Data['smoke'].corr(Data['alco'], method='spearman')

# What is the smallest age at which the number of people with CVD outnumber the number of people without CVD?
sns.countplot(x='age_years', hue='cardio', data=Data)