import warnings
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
warnings.filterwarnings("ignore")

train_original = pd.read_csv('train__titanic.csv')
test_original = pd.read_csv('test_titanic.csv')

print('Train Original Info')
print(train_original.info())
print('\n')

# Exclude some features to reduce data dimension
train = train_original.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test = test_original.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
total = [train, test]

print('Train Shape, Test Shape')
print(train.shape, test.shape)
print('\n')

# Missing cases for training set
print('Missing cases for training set')
print(train.isnull().sum())
print('\n')

# Age missing cases
print('Age Missing Cases')
print(train[train['Age'].isnull()].head())
print('\n')

# Distribution of Age, condition = Pclass
train[train.Pclass == 1].Age.plot(kind='kde', color='r', label='1st class')
train[train.Pclass == 2].Age.plot(kind='kde', color='b', label='2nd class')
train[train.Pclass == 3].Age.plot(kind='kde', color='g', label='3rd class')
plt.xlabel('Age')
plt.legend(loc='best')
plt.grid()
plt.show()
plt.close()


# Create function to replace NaN with the median value for each ticket class
def fill_missing_age(dataset):
    for i in range(1, 4):
        median_age = dataset[dataset["Pclass"] == i]["Age"].median()
        dataset["Age"] = dataset["Age"].fillna(median_age)
        return dataset


print('Fill Missing Age in Training set')
train = fill_missing_age(train)

# Embarked missing cases
print('Embarked missing cases')
print(train[train['Embarked'].isnull()])
print('\n')

# Create Barplot
sns.barplot(x="Embarked", y="Fare", hue="Sex", data=train)
plt.show()
plt.close()

# Considering Sex=female and Fare=80, Ports of Embarkation (Embarked) for two missing cases can be assumed to be Cherbourg (C).
# Replace missing cases with C
print(
    'Considering Sex=female and Fare=80, Ports of Embarkation (Embarked) for two missing cases can be assumed to be Cherbourg (C).')
print('Replace missing cases with C')
train["Embarked"] = train["Embarked"].fillna('C')
print('\n')

print('**Testing Set: Check and Impute missing cases**')

# Missing cases for testing set
print('Missing cases for testing set')
print(test.isnull().sum())
print('\n')

# Age missing cases
print('Age missing cases')
print(test[test['Age'].isnull()].head())
print('\n')

# Distribution of Age, condition = Pclass
print('Distribution of Age, condition = Pclass')
test[test.Pclass == 1].Age.plot(kind='kde', color='r', label='1st class')
test[test.Pclass == 2].Age.plot(kind='kde', color='b', label='2nd class')
test[test.Pclass == 3].Age.plot(kind='kde', color='g', label='3rd class')
plt.xlabel('Age')
plt.legend(loc='best')
plt.grid()
plt.show()
plt.close()

# Replace missing cases with the median age for each ticket class.
print('Replace missing cases with the median age for each ticket class.')
test = fill_missing_age(test)


# Create function to replace NaN with the median fare with given conditions
def fill_missing_fare(dataset):
    median_fare = dataset[(dataset["Pclass"] == 3) & (dataset["Embarked"] == "S")]["Fare"].median()
    dataset["Fare"] = dataset["Fare"].fillna(median_fare)
    return dataset


test = fill_missing_fare(test)
# Re-Check for missing cases
print('Re-Check for missing cases')
print(train.isnull().any())
print('\n')

# Boxplot for Age
sns.boxplot(x=train["Survived"], y=train["Age"])
plt.show()
plt.close()

# discretize Age feature
for dataset in total:
    dataset.loc[dataset["Age"] <= 9, "Age"] = 0
    dataset.loc[(dataset["Age"] > 9) & (dataset["Age"] <= 19), "Age"] = 1
    dataset.loc[(dataset["Age"] > 19) & (dataset["Age"] <= 29), "Age"] = 2
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[dataset["Age"] > 39, "Age"] = 4
sns.countplot(x="Age", data=train, hue="Survived")
plt.show()
plt.close()

# Boxplot for Fare
sns.boxplot(x=train["Survived"], y=train["Fare"])
plt.show()
plt.close()

# discretize Fare
print('Discrete-ize fare')
print(pd.qcut(train["Fare"], 8).value_counts())
print('\n')

for dataset in total:
    dataset.loc[dataset["Fare"] <= 7.75, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.75) & (dataset["Fare"] <= 7.91), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 9.841), "Fare"] = 2
    dataset.loc[(dataset["Fare"] > 9.841) & (dataset["Fare"] <= 14.454), "Fare"] = 3
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 24.479), "Fare"] = 4
    dataset.loc[(dataset["Fare"] > 24.479) & (dataset["Fare"] <= 31), "Fare"] = 5
    dataset.loc[(dataset["Fare"] > 31) & (dataset["Fare"] <= 69.487), "Fare"] = 6
    dataset.loc[dataset["Fare"] > 69.487, "Fare"] = 7

sns.countplot(x="Fare", data=train, hue="Survived")
plt.show()
plt.close()

# Countplot for the number of siblings/spouse
print('Countplot for the number of siblings/spouse')
sns.countplot(x="SibSp", data=train, hue="Survived")
plt.show()
plt.close()
print('\n')

# Countplot for the number of parents/childrens
print('Countplot for the number of parents/childrens')
sns.countplot(x="Parch", data=train, hue="Survived")
plt.show()
plt.close()
print('\n')

# Convert SibSp into binary feature
for dataset in total:
    dataset.loc[dataset["SibSp"] == 0, "SibSp"] = 0
    dataset.loc[dataset["SibSp"] != 0, "SibSp"] = 1

sns.countplot(x="SibSp", data=train, hue="Survived")
plt.show()
plt.close()

# Convert Parch into binary feature
for dataset in total:
    dataset.loc[dataset["Parch"] == 0, "Parch"] = 0
    dataset.loc[dataset["Parch"] != 0, "Parch"] = 1

sns.countplot(x="Parch", data=train, hue="Survived")
plt.show()
plt.close()

# Scikit learn estimators require numeric features
print('Scikit learn estimators require numeric features')
sex = {'female': 0, 'male': 1}
embarked = {'C': 0, 'Q': 1, 'S': 2}
print(sex)
print(embarked)

# Convert categorical features to numeric using mapping function
print('Convert categorical features to numeric using mapping function')
for dataset in total:
    dataset['Sex'] = dataset['Sex'].map(sex)
    dataset['Embarked'] = dataset['Embarked'].map(embarked)

print(train.head())

# total survival rate of train dataset
print('total survival rate of train dataset')
survived_cases = 0
for i in range(891):
    if train.Survived[i] == 1:
        survived_cases = survived_cases + 1

total_survival_rate = float(survived_cases) / float(891)

print('%0.4f' % (total_survival_rate))


# Survival rate under each feature condition
def survival_rate(feature):
    rate = train[[feature, 'Survived']].groupby([feature], as_index=False).mean().sort_values(by=[feature],
                                                                                              ascending=True)
    sns.factorplot(x=feature, y="Survived", data=rate)


for feature in ["Age", "Fare", "SibSp", "Parch", "Sex", "Embarked", "Pclass"]:
    survival_rate(feature)

# Inter-relationship between Fare and Pclass
sns.countplot(x="Fare", data=train, hue="Pclass")
plt.show()
plt.close()

# Relationship between Embarked and other features
train.groupby(["Embarked"], as_index=False).mean()

# Seperate input features from target feature
x = train.drop("Survived", axis=1)
y = train["Survived"]

# Split the data into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1)

# Take a look at the shape
print('Taking a look at training and testing data shape')
print(x_train.shape, y_train.shape)
print('\n')

# Decision Tree Classifier
print('Decision Tree Classifier')
clf = DecisionTreeClassifier(random_state=1)

# Run 10 fold cross validation
print('Run 10 fold cross validation')
cvs = cross_val_score(clf, x, y, cv=5)
print(cvs)

# Show cross validation score mean and std
print('Show cross validation score mean and std')
print("Accuracy: %0.4f (+/- %0.4f)" % (cvs.mean(), cvs.std() * 2))

# Fit the model with data
clf.fit(x_train, y_train)

# Accuracy
acc_decision_tree = round(clf.score(x_train, y_train), 4)
print("Accuracy: %0.4f" % (acc_decision_tree))

# Predict y given validation set
print('Predict y given validation set')
predictions = clf.predict(x_test)

# Take a look at the confusion matrix ([TN,FN],[FP,TP])
print('Take a look at the confusion matrix ([TN,FN],[FP,TP])')
print(confusion_matrix(y_test, predictions))
print('\n')

# Precision
print("Precision: %0.4f" % precision_score(y_test, predictions))
psc = precision_score(y_test, predictions)
print('The precision is %0.4f. Thus, we may conclude that %0.2f of tuples that the classifier labeled as positive are actually positive by this model.' % (psc, psc * 100))

# Recall score
print("Recall: %0.4f" % recall_score(y_test, predictions))
recl = recall_score(y_test, predictions)
print('The recall is %0.4f. Thus, we may conclude that %0.2f of real positive tuples were classified by the decision tree classifier.' % (recl, recl * 100))

# Print classification report
print('Print classification report')
print(classification_report(y_test, predictions))
print('\n')

# Get data to plot ROC Curve
fp, tp, th = roc_curve(y_test, predictions)
roc_auc = auc(fp, tp)

# Plot ROC Curve
plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
         label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.close()

# **4. Conclusion:**
#
# To sum up, passengers had higher chance of survival:
#
# - if they belonged to the first class (or hold expensive ticket)
#
# - if they were female
#
# - if they were young
#
# - if they had family
#
# - if they came from Cherbourg
#
# Among these five conditions, ticket class, sex, and age were the most influential on survival.
#
# To test the validity of the classification model, I split the "train" data into 75% of training and 25% of validation sets. And it gave us significantly high accuracy: The classification model predicted 90.27% of validation set tuples correctly. However, the prediction score was not as good as its accuracy or precision. Only 62.11% of true survival cases were detected by the classifier.
#
# We need further studies or data that would give us insight to improve the predictive model.
