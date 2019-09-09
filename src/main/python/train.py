import pandas as pd
import numpy as np
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from pypmml import Model as PyPmmlModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn2pmml.feature_extraction.text import Splitter

df = pd.read_csv("./document-classification-test/shuffled-full-set-hashed.csv", header = None)
df.columns = ['Category', 'Content']

tfidf = TfidfVectorizer(sublinear_tf=True, max_df = 0.75, min_df = 0.01, norm=None, ngram_range=(1, 2), tokenizer=Splitter())

features = tfidf.fit_transform(df['Content'].astype('U').values).toarray()
labels = df['Category']
all_categories = np.unique(labels)
print(features.shape)

N = 2
categories = []
most_corelated_unigrams = []
most_corelated_bigrams = []
for category_id in all_categories:
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    categories.append(category_id)
    most_corelated_unigrams.append(unigrams[-N:])
    most_corelated_bigrams.append(bigrams[-N:])

pd.DataFrame({'Category' : categories, 'Most corelated unigrams' : most_corelated_unigrams,
              'Most corelated bigrams' : most_corelated_bigrams})

# Topical model includes SVM, Logistic Regression and Naive Bayes
models = [
    LinearSVC(),
    LogisticRegression(),
    MultinomialNB()
]

# 5 fold cross validation
CV = 5

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


classifier = LogisticRegression(random_state=0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                 test_size=0.2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
scores=cross_val_score(classifier,X_train,y_train,cv=5)
print('accuracy_score per fold',np.mean(scores),scores)
print("accuracy_score: "+str(accuracy_score(y_test, y_pred)))


new_conf_matrix = []
for i in range(len(conf_matrix[0])):
    actual_value = sum(conf_matrix[i])
    new_conf_matrix.append([])
    for j in range(len(conf_matrix[0])):
        val = conf_matrix[i][j]
        new_conf_matrix[i].append(val / actual_value * 100)

new_conf_matrix = np.array(new_conf_matrix)

fig, ax = plt.subplots(figsize=(10,10))
category_ids = df.groupby(['Category']).count().index
sns.heatmap(new_conf_matrix.T, square=True, annot=True,fmt='d',cbar =False,
            xticklabels=category_ids, yticklabels=category_ids)
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()



# load entire dataset for training
(features_train, labels_train) = (df.loc[:, 'Content'].astype('U').values, df.loc[:, 'Category'])


### TfidfVectorizer
tfidfv = TfidfVectorizer(ngram_range=(1, 2), min_df=0.01, max_df=0.75, norm=None, tokenizer=Splitter())

## Selector.
selector = SelectKBest(chi2, k=1000)

lr = LogisticRegression(penalty='l2', dual=False,
                        tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                        class_weight='balanced', random_state=None, solver='newton-cg',  max_iter=100, multi_class='multinomial', verbose=0,
                        warm_start=False, n_jobs=-1, l1_ratio=None)

pipeline = Pipeline([('vect', tfidfv),
                     ('selector', selector),
                     ('lr', lr),
                     ])

pipeline.fit(features_train, labels_train)
from sklearn.externals import joblib
joblib.dump(pipeline, 'model.pkl')


pipeline = PMMLPipeline([('vect', tfidfv),
                         ('selector', selector),
                         ('lr', lr),
                         ])

sklearn2pmml(pipeline, 'doc_classify.pmml', with_repr=True)

model = PyPmmlModel.fromFile('doc_classify.pmml')
result = model.predict([df.iloc[13605]['Content']])





