import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = 'train.csv'
titanic_data= pd.read_csv(df)
print(df)

print(titanic_data.head())

print(titanic_data.info())

print(titanic_data.describe())

print(titanic_data['Sex'].value_counts())

print(titanic_data['Pclass'].value_counts())

print(titanic_data['Embarked'].value_counts())

plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# survival by passenger class
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# age group
titanic_data['AgeGroup'] = pd.cut(titanic_data['Age'], bins=[0, 12, 18, 35, 60, 80], labels=['Child', 'Teenager', 'Adult', 'Middle-Aged', 'Senior'])
plt.figure(figsize=(10, 6))
sns.countplot(x='AgeGroup', hue='Survived', data=titanic_data)
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked', hue='Survived', data=titanic_data)
plt.title('Survival Rate by Embarkation Point')
plt.xlabel('Embarkation Point')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = titanic_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Titanic Dataset')
plt.show()