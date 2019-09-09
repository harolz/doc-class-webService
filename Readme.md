# [Doc Classification Web Service Container

 A container that hosts static web page and RESTful APIs, which implement features of real-time, batch prediction and inference pipeline use-cases. 

## Introduction

The project mainly consists of the following components:

- Html/JavaScript based UI/UX design utilizing one-way data binding and AJAX to restful APIs
- Python/Notebook based modeling training using Scikit_learn toolkit and AWS SageMaker
- Spring Boot based RESTful API Server and static web hosting
- Docker based AWS container registry service and hosting service ECR and ECS

A typical workflow can be summarized as follows:

1. Use Notebook/Python to train a model.
2. Use command line tool provided by [sklearn2pmml](https://github.com/jpmml/sklearn2pmml) to convert a .pkl file to a pmml file
3. package source Java code into a jar which serve as invocation of model prediction model through restful API calls and static web content hosting include HTML/CSS and JavaScript  
4. build a docker image that contains the jar executable and expose port 8080
5. push the docker image to AWS ECR and deploy it with AWS ECS

## Steps

### Training Model using Scikit_learn Toolkit

Loading data to a `pandas.DataFrame` object and visualize the data distribution:

```python
import pandas
df = pandas.read_csv("shuffled-full-set-hashed.csv")
df.columns = ['Category', 'Content']
#  Print the first 20 data points -- the head of the dataset
print(df.head(20))
#  Use the describe function to describe some of the 
#  statistical properties of the data.
print(df.describe())
#  Use the groupby method to determine the class distribution
print(df.groupby('Category', sort=True).size())
cnt_categories = doc['Category'].value_counts(ascending=True)
plt.figure(figsize=(12,4))
g = sns.barplot(cnt_categories.index, cnt_categories.values, alpha=0.9)
plt.ylabel('Count', fontsize=14)
plt.xlabel('Category', fontsize=14)
plt.xticks(rotation=270)
plt.show()
```

Setting parameters for TfidfVectorizer. Term frequency from lower than 0.75 and higher than 0.01 will be accepted for vectorization. Single term and consecutive two terms will be considered. 

```python
tfidf = TfidfVectorizer(sublinear_tf=True, max_df = 0.75, min_df = 0.01, norm=None, ngram_range=(1, 2), tokenizer=Splitter())
features = tfidf.fit_transform(df['Content'].astype('U').values).toarray()
labels = df['Category']
all_categories = np.unique(labels)
print(features.shape)
```

For each category, create

```python
from sklearn_pandas import DataFrameMapper
    for fold_idx, accuracy in enumerate(accuracies):

```



First, creating a `sklearn_pandas.DataFrameMapper` object, which performs **column-oriented** feature engineering and selection work:

```python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn2pmml.decoration import ContinuousDomain

column_preprocessor = DataFrameMapper([
    (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [ContinuousDomain(), StandardScaler()])
])
```

Second, creating `Transformer` and `Selector` objects, which perform **table-oriented** feature engineering and selection work:

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn2pmml import SelectorProxy

table_preprocessor = Pipeline([
	("pca", PCA(n_components = 3)),
	("selector", SelectorProxy(SelectKBest(k = 2)))
])
```

Please note that stateless Scikit-Learn selector objects need to be wrapped into an `sklearn2pmml.SelectprProxy` object.

Third, creating an `Estimator` object:

```python
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(min_samples_leaf = 5)
```

Combining the above objects into a `sklearn2pmml.pipeline.PMMLPipeline` object, and running the experiment:

```python
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
    ("columns", column_preprocessor),
    ("table", table_preprocessor),
    ("classifier", classifier)
])
pipeline.fit(iris_X, iris_y)
```

Embedding model verification data:

```python
pipeline.verify(iris_X.sample(n = 15))
```

Storing the fitted `PMMLPipeline` object in `pickle` data format:

```python
from sklearn.externals import joblib

joblib.dump(pipeline, "pipeline.pkl", compress = 9)
```

Please see the test script file [main.py](https://github.com/jpmml/jpmml-sklearn/blob/master/src/test/resources/main.py) for more classification (binary and multi-class) and regression workflows.

### The JPMML-SkLearn side of operations

Converting the pipeline pickle file `pipeline.pkl.z` to a PMML file `pipeline.pmml`:

```java
java -jar target/jpmml-sklearn-executable-1.5-SNAPSHOT.jar --pkl-input pipeline.pkl --pmml-output pipeline.pmml
```

Getting help:

```
java -jar target/jpmml-sklearn-executable-1.5-SNAPSHOT.jar --help
```

# How to reproduce locally

1. Clone the project

```
git clone https://github.com/spicoflorin/aws-sagemaker-example.git
```

2. In the directory doc-class-webService, run:

```
  mvn clean package
```

3. build the docker image according to name customs for AWS ECR   

```
docker build . --tag your-accountid.dkr.ecr.your-region.amazonaws.com/your-ecr-repository-name
```

4. Test the image

   4.1. Start the container 

```
docker run -p 8080:8080 your-accountid.dkr.ecr.your-region.amazonaws.com/your-ecr-repository-name
```

​	   4.2. Test with post request

```
curl -X POST http://localhost:8080/predict -d '{"words":"aa1ef5f5355f 5948001254b3 ddcfb3277250 f95d0bea231b 9e0c01b8b857 e259a56993f4 9a87e8f4bd5c aa1ef5f5355f f95d0bea231b 35a437bf20ea 39995776c91f 6a95ce91efbd 174679aec918 3a918f3d2d81 43af6db29054 dc83f2b00468 43af6db29054 1df053306286 aa3c127bfd67 8f75273e5510 036087ac04f9 aa1ef5f5355f b136f6349cf3 d38820625542 d0b9e33388a7 f9f90c2328ed ebe6b1fb3a5b 2356cc591755 e2d8c082f942 8b6c5bb157a9 b590e4634f73 df3e4be4e03a 6eb1a319229d 036087ac04f9 b136f6349cf3 a1e5904d04f5 9cdf4a63deb0 f49ab97a086c c5dcd74b40a9 7e8d554779a7 56fa56419ad0 46a0c35e916c d9ef68daef4c a3360a4991fa 133d46f7ed38 0f12fd1c2b99 31fd3123f41c 33630ee5f812 586242498a88 d38820625542 bb0e7ae8fdbf d38820625542 d38820625542 6bf9c0cb01b4 1a0b71f9f7ff a0c020166d79 8d87346febd0 5d114661f44f 63b82ee0a4d8 174679aec918 3a918f3d2d81 c913f5129fe2 dc83f2b00468 377a21b394dc 1df053306286 454513214d62 8137cf3679d9 192707ed58ee 4c1f76b16699 7bfba3bd67d7 10520efc20b5 7e07ecc160fd 769e648b85f2 aecab1ec7a6d aa3c127bfd67 8e1e192ac432 6101ed18e42f 036087ac04f9 0f1b8403e5ee b136f6349cf3 036087ac04f9 7ec02e30a5b3 b136f6349cf3 b28b5d7829ea bfed1d086a6c 6ce6cc5a3203 798fe9915030 71c48df544b0 04e4d0d2f1c7 9cdf4a63deb0 f9f90c2328ed a3b334c6eefd ebe6b1fb3a5b 8e93a2273a93 9bc65adc033c 9ad186d42f69 4ffb12504ac6 6b343f522f78 0662d94b3d3b fe64d3cdfe5b 2519927ae3fa 74586e35c44a 6365c4563bd1 4ffb12504ac6 2ea49cf89745 e504ee0aaf6d 6ce6cc5a3203 af671fbeb212 957b5cf4e65e 99716e581500 056314258a60 056314258a60 997c023f641e 6b1c8f75a7e2 4e5019f629a9 cafaf222091d cafaf222091d f7ae6f8257da 04503bc22789 ebbd827fe2a0 6ca2dd348663 2fb652bf7937 7d9e333a86da 7ec02e30a5b3 9cdf4a63deb0 b28b5d7829ea 0c37b8e00f96 bfed1d086a6c 1160c83555d2 04503bc22789 bd50a6b2259f cfd22ba194a9 ed5d3a65ee2d f2f35d4c0c22 55c0cd1cc044 30ca33997a38 84bf8a94981c 8eedff84f18d b208ae1e8232 4eb7c8207490 454513214d62 8137cf3679d9 6365c4563bd1 4c1f76b16699 9a87e8f4bd5c 9f11111004ec 56a0a522a4dd c337a85b8ef9 9b88c973ae02 422068f04236 6bf9c0cb01b4 e94953618947 1b6d0614f2c7 afb1e3806fc1 f2f35d4c0c22 036087ac04f9 6bf9c0cb01b4 e94953618947 c337a85b8ef9 b136f6349cf3 cafaf222091d a1fde4983c10 2ed97f462806 c8f5ad40a683 8f75273e5510 9e0c01b8b857 7d9e333a86da f4b04aeadc5e 9a87e8f4bd5c 2556150a673a b9699ce57810 133d46f7ed38 04503bc22789 b9699ce57810 e7e059c82399 ce1f034abb5d 6f40fa36485c 7e07ecc160fd 6ef2ade170d9 769e648b85f2 d9ef68daef4c 6ce6cc5a3203 b513aeae3f9a 30ca33997a38 769e648b85f2 ec3406979928 9fbc5a0e2daf bf3aa3fc66f6 e4dad7cb07b6 fc25f79e6d18 e1b9e4df3a88 2556150a673a 878460b4304e 7e07ecc160fd 586242498a88 769e648b85f2 586242498a88 e4dad7cb07b6 9431856ec97e 6bf9c0cb01b4 e94953618947 c337a85b8ef9 094453b4e4ae c33b5c3d0449 b9699ce57810 489eaf3a08fb 95ef80a0b841 f0666bdbc8a5 d9ef68daef4c 56c2c356d772 e7e2fc1908c0 b2c1fd62c2ac eb52c980ed38 e162da38a9d7 cb7631b88e51 8ebb4fffd292 a3360a4991fa f4b04aeadc5e 422068f04236 d38820625542 133d46f7ed38 1015893e384a 97b6014f9e50 470aa9b28443 29455ef44c25 6b343f522f78 29e88482be15 2d00e7e4d33f 5c02c2aaa67b 6b343f522f78 b008843106fd d38820625542 636540642f5d 133d46f7ed38 1a0b71f9f7ff 890ad17d1696 0cbca93be301 b208ae1e8232 b008843106fd ba3a3713691e bf15989af17d 4f5e0215c1bf d63be9e66da8 5c2db045bc17 d38820625542 7d9e333a86da d8535c18626a 0cbca93be301 d38820625542 9e0c01b8b857"}' -H "Content-Type: application/json"
```

Web Service returns JSON response.

```
{"result":"BINDER","confidence":0.6880633254455778}
```

​		4.3. Test with GET Request

```
$curl -G http://localhost:8080/predict/aa1ef5f5355f%205948001254b3%20ddcfb3277250%20f95d0bea231b%209e0c01b8b857%20e259a56993f4%209a87e8f4bd5c%20aa1ef5f5355f%20f95d0bea231b%2035a437bf20ea%2039995776c91f%206a95ce91efbd%20174679aec918%203a918f3d2d81%2043af6db29054%20dc83f2b00468%2043af6db29054%201df053306286%20aa3c127bfd67%208f75273e5510%20036087ac04f9%20aa1ef5f5355f%20b136f6349cf3%20d38820625542%20d0b9e33388a7%20f9f90c2328ed%20ebe6b1fb3a5b%202356cc591755%20e2d8c082f942%208b6c5bb157a9%20b590e4634f73%20df3e4be4e03a%206eb1a319229d%20036087ac04f9%20b136f6349cf3%20a1e5904d04f5%209cdf4a63deb0%20f49ab97a086c%20c5dcd74b40a9%207e8d554779a7%2056fa56419ad0%2046a0c35e916c%20d9ef68daef4c%20a3360a4991fa%20133d46f7ed38%200f12fd1c2b99%2031fd3123f41c%2033630ee5f812%20586242498a88%20d38820625542%20bb0e7ae8fdbf%20d38820625542%20d38820625542%206bf9c0cb01b4%201a0b71f9f7ff%20a0c020166d79%208d87346febd0%205d114661f44f%2063b82ee0a4d8%20174679aec918%203a918f3d2d81%20c913f5129fe2%20dc83f2b00468%20377a21b394dc%201df053306286%20454513214d62%208137cf3679d9%20192707ed58ee%204c1f76b16699%207bfba3bd67d7%2010520efc20b5%207e07ecc160fd%20769e648b85f2%20aecab1ec7a6d%20aa3c127bfd67%208e1e192ac432%206101ed18e42f%20036087ac04f9%200f1b8403e5ee%20b136f6349cf3%20036087ac04f9%207ec02e30a5b3%20b136f6349cf3%20b28b5d7829ea%20bfed1d086a6c%206ce6cc5a3203%20798fe9915030%2071c48df544b0%2004e4d0d2f1c7%209cdf4a63deb0%20f9f90c2328ed%20a3b334c6eefd%20ebe6b1fb3a5b%208e93a2273a93%209bc65adc033c%209ad186d42f69%204ffb12504ac6%206b343f522f78%200662d94b3d3b%20fe64d3cdfe5b%202519927ae3fa%2074586e35c44a%206365c4563bd1%204ffb12504ac6%202ea49cf89745%20e504ee0aaf6d%206ce6cc5a3203%20af671fbeb212%20957b5cf4e65e%2099716e581500%20056314258a60%20056314258a60%20997c023f641e%206b1c8f75a7e2%204e5019f629a9%20cafaf222091d%20cafaf222091d%20f7ae6f8257da%2004503bc22789%20ebbd827fe2a0%206ca2dd348663%202fb652bf7937%207d9e333a86da%207ec02e30a5b3%209cdf4a63deb0%20b28b5d7829ea%200c37b8e00f96%20bfed1d086a6c%201160c83555d2%2004503bc22789%20bd50a6b2259f%20cfd22ba194a9%20ed5d3a65ee2d%20f2f35d4c0c22%2055c0cd1cc044%2030ca33997a38%2084bf8a94981c%208eedff84f18d%20b208ae1e8232%204eb7c8207490%20454513214d62%208137cf3679d9%206365c4563bd1%204c1f76b16699%209a87e8f4bd5c%209f11111004ec%2056a0a522a4dd%20c337a85b8ef9%209b88c973ae02%20422068f04236%206bf9c0cb01b4%20e94953618947%201b6d0614f2c7%20afb1e3806fc1%20f2f35d4c0c22%20036087ac04f9%206bf9c0cb01b4%20e94953618947%20c337a85b8ef9%20b136f6349cf3%20cafaf222091d%20a1fde4983c10%202ed97f462806%20c8f5ad40a683%208f75273e5510%209e0c01b8b857%207d9e333a86da%20f4b04aeadc5e%209a87e8f4bd5c%202556150a673a%20b9699ce57810%20133d46f7ed38%2004503bc22789%20b9699ce57810%20e7e059c82399%20ce1f034abb5d%206f40fa36485c%207e07ecc160fd%206ef2ade170d9%20769e648b85f2%20d9ef68daef4c%206ce6cc5a3203%20b513aeae3f9a%2030ca33997a38%20769e648b85f2%20ec3406979928%209fbc5a0e2daf%20bf3aa3fc66f6%20e4dad7cb07b6%20fc25f79e6d18%20e1b9e4df3a88%202556150a673a%20878460b4304e%207e07ecc160fd%20586242498a88%20769e648b85f2%20586242498a88%20e4dad7cb07b6%209431856ec97e%206bf9c0cb01b4%20e94953618947%20c337a85b8ef9%20094453b4e4ae%20c33b5c3d0449%20b9699ce57810%20489eaf3a08fb%2095ef80a0b841%20f0666bdbc8a5%20d9ef68daef4c%2056c2c356d772%20e7e2fc1908c0%20b2c1fd62c2ac%20eb52c980ed38%20e162da38a9d7%20cb7631b88e51%208ebb4fffd292%20a3360a4991fa%20f4b04aeadc5e%20422068f04236%20d38820625542%20133d46f7ed38%201015893e384a%2097b6014f9e50%20470aa9b28443%2029455ef44c25%206b343f522f78%2029e88482be15%202d00e7e4d33f%205c02c2aaa67b%206b343f522f78%20b008843106fd%20d38820625542%20636540642f5d%20133d46f7ed38%201a0b71f9f7ff%20890ad17d1696%200cbca93be301%20b208ae1e8232%20b008843106fd%20ba3a3713691e%20bf15989af17d%204f5e0215c1bf%20d63be9e66da8%205c2db045bc17%20d38820625542%207d9e333a86da%20d8535c18626a%200cbca93be301%20d38820625542%209e0c01b8b857
```

# Push to ECR

1. Login to ECR

```
$(aws ecr get-login --region your-region --no-include-email)
```

2. Create ECR repository "your-ecr-repository-name"

```
aws ecr create-repository --repository-name "your-ecr-repository-name"
```

3. Push local docker image to ECR

```
docker push your-accountid.dkr.ecr.your-region.amazonaws.com/your-ecr-repository-name
```