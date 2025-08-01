# Databricks notebook source
# MAGIC %md # Read only experiment issue - use case 1
# MAGIC With manual `create_experiment`code.
# MAGIC It works but it's not the standard which is to use `autologging`feature
# MAGIC
# MAGIC

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md ## Read the data
# MAGIC Read the white wine quality and red wine quality CSV datasets and merge them into a single DataFrame.

# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md ## Preprocess data
# MAGIC Before training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the dataset to train a baseline model
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = data.drop(["quality"], axis=1)
y = data.quality

# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Issue is there!
# MAGIC Manually create an experiment under "/user"

# COMMAND ----------

exp_id = mlflow.create_experiment(
    name="/Workspace/Users/yann.chaput@axa.com/experiments/mlflow_exp_logging_issue_manual"
)
print(exp_id)

# COMMAND ----------

# MAGIC %md ## Train a baseline model
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC
# MAGIC Build a simple classifier using scikit-learn and use MLflow to keep track of the model's accuracy and save the model for later use.

# COMMAND ----------

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

experiment = mlflow.set_experiment(experiment_id=exp_id)

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='untuned_random_forest'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)

# COMMAND ----------

# MAGIC %md Load the model into a Spark UDF, so it can be applied to the Delta table.

# COMMAND ----------

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}@Champion")

# COMMAND ----------

# Read the "new data" from the Unity Catalog table

new_data = spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_data")

# COMMAND ----------

from pyspark.sql.functions import struct

# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

# Each row now has an associated prediction. Note that the xgboost function does not output probabilities by default, so the predictions are not limited to the range [0, 1].
display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve the model
# MAGIC To productionize the model for low latency predictions, use Mosaic AI Model Serving to deploy the model to an endpoint. The following cell shows how to use the MLflow Deployments SDK to create a model serving endpoint.   
# MAGIC
# MAGIC For more information about Mosaic AI Model Serving, including how to use the UI or REST API to create a serving endpoint, see the documentation: [AWS](https://docs.databricks.com/machine-learning/model-serving/index.html) | [Azure](https://docs.microsoft.com/azure/databricks/machine-learning/model-serving/index) | [GCP](https://docs.gcp.databricks.com/en/machine-learning/model-serving/index.html). 

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
endpoint = client.create_endpoint(
    name="wine-model-endpoint",
    config={
        "served_entities": [
            {
                "name": "wine-entity",
                "entity_name": model_name,
                "entity_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ],
      }
)
