import pandas as pd

df = pd.read_json(
    "Computing_requirements.json")  # Reading json file to see if the user's input have been proofed or not.
df.info()

df['intents'].value_counts()  # Displays value for that column

df.head()

df[df['intents'] == 1].head()  # Executes malicious data compared to genuine.

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')


def plot_attribute_countplot_by_label(dataset, attribute_name, value_list=None):
    if value_list is None:
        value_list = dataset[attribute_name].unique()  # Converts the malicious data to genuine if not equals 0
    fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(14, len(value_list) * 2))
    sns.countplot(y='intents', hue=attribute_name, hue_order=value_list,
                  data=dataset[dataset['intents'] == 0], ax=axis1)
    sns.countplot(y='intents', hue=attribute_name, hue_order=value_list,
                  data=dataset[dataset['intents'] == 1], ax=axis2)  # This is the value when the data in genuine

plot_attribute_countplot_by_label(df, "http_version")

from sklearn.model_selection import train_test_split

# Splits the train and test into subnets, adding up the training set will be 80% and the test set will be 20%
attributes = ['uri', 'is_static', 'http_version', 'has_referer', 'method']
x_train, x_test, y_train, y_test = train_test_split(df[attributes], df['intents'], test_size=0.2,
                                                    stratify=df['intents'], random_state=0)

x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2,
                                                  stratify=y_train, random_state=0)
print('Train:', len(y_train), 'Dev:', len(y_dev), 'Test', len(y_test))

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(analyzer='char', min_df=10)
n_grams_train = count_vectorizer.fit_transform(x_train['uri'])
n_grams_dev = count_vectorizer.transform(x_dev['uri'])

print('Number of features: ', len(count_vectorizer.vocabulary_))

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier(random_state=0)
sgd.fit(n_grams_train, y_train)
y_pred_sgd = sgd.predict(n_grams_dev)
print("SGDClassifier accuracy:", accuracy_score(y_dev, y_pred_sgd))

from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(n_grams_train, y_train)
print("DummyClassifier accuracy:", dummy_clf.score(n_grams_dev, y_dev))

from sklearn.metrics import precision_score, recall_score
print('Precision:', precision_score(y_dev, y_pred_sgd))
print('Recall:', recall_score(y_dev, y_pred_sgd))

from sklearn.metrics import precision_recall_curve
from ggplot import ggplot, aes, geom_line
y_pred_scores = sgd.decision_function(n_grams_dev)

def plot_precision_recall_curve(y_true, y_pred_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_scores)
    return ggplot(aes(x='recall', y='precision'),
                  data=pd.DataFrame({"precision": precision, "recall": recall})) + geom_line()

plot_precision_recall_curve(y_dev, y_pred_scores)
