import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the adult data set downloaded from UCI
Data = pd.read_csv("F:/Deep Learning/mlcourse_open/adult.data.csv")

# 1. How many men and women (sex feature) are represented in this dataset?
Data['sex'].value_counts()

# 2. What is the average age (age feature) of women?
Data[Data['sex'] == 'Female'].age.mean()

# 3. What is the percentage of German citizens (native-country feature)?
Data['native-country'].value_counts(normalize=True)*100

# 4. Make a population histogram (bar plot) of people's education (education feature). What are the mean and standard deviation of age for those who earn more than 50K per year (salary feature)?
Data[Data['salary'] == '>50K'].age.mean()
Data[Data['salary'] == '>50K'].age.std()

# 5. What are the mean and standard deviation of age for those who earn less than 50K per year?
Data[Data['salary'] == '<=50K'].age.mean()
Data[Data['salary'] == '<=50K'].age.std()

# 6. Is it true that people who earn more than 50K have at least high school education? (education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters or Doctorate feature)
Data[Data['salary'] == '>50K'].education.value_counts()

# 7. Find the maximum age of men of Amer-Indian-Eskimo race
Data[Data['race'] == 'Amer-Indian-Eskimo'].age.max()

# 8. Among whom is the proportion of those who earn a lot (>50K) greater: married or single men (marital-status feature)? Consider as married those who have a marital-status starting with Married (Married-civ-spouse, Married-spouse-absent or Married-AF-spouse), the rest are considered bachelors.
pd.crosstab(Data['marital-status'],Data['salary'],normalize=True)*100

# 9. What is the maximum number of hours a person works per week (hours-per-week feature)? How many people work such a number of hours, and what is the percentage of those who earn a lot (>50K) among them?
Data['hours-per-week'].max()
Data[Data['hours-per-week'] == 99].count()

# 10. Count the average time of work (hours-per-week) for those who earn a little and a lot (salary) for each country (native-country). What will these be for Japan?
Japan = Data[Data['native-country'] == 'Japan']
pd.pivot_table(data=Data, index = 'native-country', columns='salary', values='hours-per-week',aggfunc='mean')