# Doc Classification Web Service Container

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

Loading data to a `pandas.DataFrame` object and observe the data:

```python
import pandas
df = pandas.read_csv("shuffled-full-set-hashed.csv")
df.columns = ['Category', 'Content']
#  Print the first 20 data points -- the head of the dataset
print(doc.head(20))
#  Use the describe function to describe some of the 
#  statistical properties of the data.
print(doc.describe())

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

joblib.dump(pipeline, "pipeline.pkl.z", compress = 9)
```

Please see the test script file [main.py](https://github.com/jpmml/jpmml-sklearn/blob/master/src/test/resources/main.py) for more classification (binary and multi-class) and regression workflows.

### The JPMML-SkLearn side of operations

Converting the pipeline pickle file `pipeline.pkl.z` to a PMML file `pipeline.pmml`:

```
java -jar target/jpmml-sklearn-executable-1.5-SNAPSHOT.jar --pkl-input pipeline.pkl.z --pmml-output pipeline.pmml
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

2. In the directory aws-sagemaker-example, run:

```
  mvn clean package
```

3. In the directory aws-sagemaker-example, build the docker image in the format of ECR Sagemaker  

```
docker build . --tag your-accountid.dkr.ecr.your-region.amazonaws.com/your-ecr-repository-name
```

4. Test the image

   4.1. Start the container 

```
docker run -p 8080:8080 your-accountid.dkr.ecr.your-region.amazonaws.com/your-ecr-repository-name
```

​      4.2 Test with a value

```
curl -X POST http://localhost:8080/invocations -d '6.7,2.5,5.8,1.8,Iris-virginica' -H 'Content-Type: text/csv'
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