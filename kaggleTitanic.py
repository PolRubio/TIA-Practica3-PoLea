#Script Titanic
#Eina Original Jupyter NOTEBOOK

import pandas as pd

#Carreguem les dades

train_data = pd.read_csv("train.csv")
print("Anàlisi Train Data")
train_data.info()

test_data = pd.read_csv("test.csv")
print("Anàlisi Test Data")

submission = pd.read_csv("submission.csv")
print("Survived de test")

test_data = pd.merge(test_data, submission, on = 'PassengerId', how = 'left')

test_data.info()

#Les unim per fer analítica descriptiva
#axis = 0 concatena per files
dfComplet = pd.concat([train_data, test_data], axis=0)

print("Anàlisi Set de Dades Complert")
dfComplet.info()

#Fem inspecció visual de les dades
dfComplet.head(5)

#Eliminem les columnes que no ens aporten informació
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin','Embarked'], axis = 1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin','Embarked'], axis = 1)

dfComplet = dfComplet.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis = 1)

#Eliminem els NA de train i test, però no del dataframe d'analítica descriptiva
train_data = train_data.dropna()
test_data = test_data.dropna()

#Com eren els passatgers del Titanic?
#Visualitzar la distribució de les edats

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(dfComplet['Age'], kde=True, color='skyblue')
plt.title('Distribució d\'edats dels passatgers')
plt.xlabel('Edat')
plt.ylabel('Nombre')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(dfComplet['Sex'], color='skyblue')
plt.title('Distribució per Gènere')
plt.xlabel('Gènere')
plt.ylabel('Nombre')
plt.show()

def transformarClasse (x):
    if x == 1:
        return '1era Classe'
    elif x == 2:
        return '2ona Classe'
    elif x == 3:
        return '3era Classe'
    else:
        return "No informat"
    
dfComplet['Pclass'] = dfComplet['Pclass'].transform(transformarClasse)

plt.figure(figsize=(10, 6))
sns.histplot(dfComplet['Pclass'], color='skyblue')
plt.title('Distribució per Classe')
plt.xlabel('Classe')
plt.ylabel('Nombre')
plt.show()

def transformarSupervivent (x):
    if x == 1:
        return 'Supervivent'
    elif x == 0:
        return 'No supervivent'
    else:
        return "No informat"
    
dfComplet['Survived'] = [transformarSupervivent(i) for i in dfComplet['Survived']]
plt.figure(figsize=(10, 6))
sns.histplot(dfComplet['Survived'], color='skyblue')
plt.title('Distribució dels Supervivents')
plt.xlabel('Supervivents')
plt.ylabel('Nombre')
plt.show()

#Distribució per edat del Supervivents

dfCompletSupervivents = dfComplet[dfComplet['Survived']=='Supervivent']

plt.figure(figsize=(10, 6))
sns.histplot(dfCompletSupervivents['Age'], kde=True, color='skyblue')
plt.title('Distribució d\'edats dels Supervivents')
plt.xlabel('Edat')
plt.ylabel('Nombre')
plt.show()

#Distribució per edat dels No Supervivents

dfCompletNoSupervivents = dfComplet[dfComplet['Survived']=='No supervivent']

plt.figure(figsize=(10, 6))
sns.histplot(dfCompletNoSupervivents['Age'], kde=True, color='skyblue')
plt.title('Distribució d\'edat dels No Supervivents')
plt.xlabel('Edat')
plt.ylabel('Nombre')
plt.show()

#Distribució per gènere del Supervivents

plt.figure(figsize=(10, 6))
sns.histplot(dfCompletSupervivents['Sex'], color='skyblue')
plt.title('Distribució per Gènere dels Supervivents')
plt.xlabel('Gènere')
plt.ylabel('Nombre')
plt.show()

#Distribució per gènere del No Supervivents

plt.figure(figsize=(10, 6))
sns.histplot(dfCompletNoSupervivents['Sex'], color='skyblue')
plt.title('Distribució per Gènere dels No Supervivents')
plt.xlabel('Gènere')
plt.ylabel('Nombre')
plt.show()

#Distribució per Classe del Supervivents

plt.figure(figsize=(10, 6))
sns.histplot(dfCompletSupervivents['Pclass'], color='skyblue')
plt.title('Distribució per Classe dels Supervivents')
plt.xlabel('Classe')
plt.ylabel('Nombre')
plt.show()

#Distribució per Classe i Gènere del Supervivents

pd.crosstab(index=dfCompletSupervivents['Sex'],columns=dfCompletSupervivents['Pclass']).plot(kind='bar')

#Distribució per Classe del No Supervivents

plt.figure(figsize=(10, 6))
sns.histplot(dfCompletNoSupervivents['Pclass'], color='skyblue')
plt.title('Distribució per Classe dels No Supervivents')
plt.xlabel('Classe')
plt.ylabel('Nombre')
plt.show()

#Distribució per Classe i Gènere del No Supervivents

pd.crosstab(index=dfCompletNoSupervivents['Sex'],columns=dfCompletNoSupervivents['Pclass']).plot(kind='bar')

#Com es distribueixen les Classes

def transformarEdat (x):
    if x > -1 and x < 13:
        return 'Infant'
    elif x > 12 and x < 18:
        return 'Adolescent'
    elif x > 18 and x < 25:
        return 'Jove'
    elif x > 24 and x < 40:
        return 'Adult Jove'
    elif x > 39 and x < 65:
        return 'Adult Sènior'
    elif x > 64:
        return '3era edat'
    else:
        return "No informat"
    
dfComplet['Age'] = [transformarEdat(i) for i in dfComplet['Age']]

dfComplet1era = dfComplet[dfComplet['Pclass']=='1era Classe']
dfComplet2ona = dfComplet[dfComplet['Pclass']=='2ona Classe']
dfComplet3era = dfComplet[dfComplet['Pclass']=='3era Classe']

pd.crosstab(index=dfComplet1era['Sex'],columns=dfComplet1era['Age']).plot(kind='bar')

pd.crosstab(index=dfComplet2ona['Sex'],columns=dfComplet2ona['Age']).plot(kind='bar')

pd.crosstab(index=dfComplet3era['Sex'],columns=dfComplet3era['Age']).plot(kind='bar')

def transformarPort (x):
    if x == 'S':
        return 'Southhampton'
    elif x == 'C':
        return 'Cherburgo'
    elif x == 'Q':
        return 'Queenstown'
    else:
        return "No informat"

dfComplet['Embarked'] = [transformarPort(i) for i in dfComplet['Embarked']]    
    
dfComplet.groupby(['Embarked']).count()

pd.crosstab(index=dfComplet['Embarked'],columns=dfComplet['Pclass']).plot(kind='bar')

#Aplicarem KNN

def transformarGenere (x):
    if x == 'male':
        return 0
    else:
        return 1

train_data['Sex'] = [transformarGenere(i) for i in train_data['Sex']]   
test_data['Sex'] = [transformarGenere(i) for i in test_data['Sex']] 

y_train = train_data['Survived']
x_train = train_data.drop(['Survived'], axis=1)

y_test = test_data['Survived']
x_test = test_data.drop(['Survived'], axis=1)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(x_train,y_train)

predictionsKNN = knn.predict(x_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("KNN :",accuracy_score(y_test,predictionsKNN))
print(confusion_matrix(y_test,predictionsKNN))
print(classification_report(y_test,predictionsKNN))