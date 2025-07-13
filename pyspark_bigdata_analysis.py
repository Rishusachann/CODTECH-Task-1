
# 📌 Step 1: Initialize Spark Session
from pyspark.sql import SparkSession

spark = SparkSession.builder     .appName("Big Data Analysis - Boston Housing")     .getOrCreate()

# Step 2: Load the Dataset
df = spark.read.csv("boston.csv", header=True, inferSchema=True)

# Step 3: Basic Exploration
df.printSchema()
df.show(5)
df.describe().show()

# Step 4: Check for Missing Values
from pyspark.sql.functions import col, sum
df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()

# Step 5: Correlation Matrix (e.g., with target column 'MEDV')
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=[c for c in df.columns if c != 'MEDV'], outputCol="features")
vec_df = assembler.transform(df).select("features")
correlation = Correlation.corr(vec_df, "features").head()
print("Correlation Matrix:\n", correlation[0])

# Step 6: Feature Engineering - Assemble Features
feature_cols = [c for c in df.columns if c != "MEDV"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", "MEDV")

# Step 7: Scaling Features
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(data)
scaled_data = scaler_model.transform(data).select("scaled_features", "MEDV")

# Step 8: PCA for Dimensionality Reduction
from pyspark.ml.feature import PCA
pca = PCA(k=5, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(scaled_data)
pca_result = pca_model.transform(scaled_data).select("pca_features", "MEDV")

# Step 9: Train/Test Split (PCA)
train_pca, test_pca = pca_result.randomSplit([0.8, 0.2], seed=42)

# Step 10: Linear Regression using PCA Features
from pyspark.ml.regression import LinearRegression
lr_pca = LinearRegression(featuresCol="pca_features", labelCol="MEDV")
lr_model_pca = lr_pca.fit(train_pca)
eval_pca = lr_model_pca.evaluate(test_pca)
print("PCA - RMSE:", eval_pca.rootMeanSquaredError)
print("PCA - R²:", eval_pca.r2)

# Step 11: SQL-Based Querying
df.createOrReplaceTempView("boston")
spark.sql("""
    SELECT ROUND(RM) AS avg_rooms, COUNT(*) AS count, ROUND(AVG(MEDV), 2) AS avg_price
    FROM boston
    GROUP BY ROUND(RM)
    ORDER BY avg_rooms
""").show()

# Step 12: Caching for Performance
data.cache()
data.count()

# Step 13: Train/Test Split (Original Data)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 14: Linear Regression on Original Features
lr = LinearRegression(featuresCol="features", labelCol="MEDV")
lr_model = lr.fit(train_data)
test_results = lr_model.evaluate(test_data)
print("Original - RMSE:", test_results.rootMeanSquaredError)
print("Original - R²:", test_results.r2)

# Step 15: Cross-Validation
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

param_grid = ParamGridBuilder()     .addGrid(lr.regParam, [0.01, 0.1])     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])     .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=param_grid,
                          evaluator=RegressionEvaluator(labelCol="MEDV"),
                          numFolds=5)

cv_model = crossval.fit(train_data)
cv_results = cv_model.bestModel.evaluate(test_data)
print("CV Best Model RMSE:", cv_results.rootMeanSquaredError)
print("CV Best Model R²:", cv_results.r2)

# Step 16: Random Forest Regressor
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(featuresCol="features", labelCol="MEDV", numTrees=100)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)
rf_eval = RegressionEvaluator(labelCol="MEDV", predictionCol="prediction", metricName="r2").evaluate(rf_predictions)
print("Random Forest R²:", rf_eval)

# Step 17: Stop Spark Session
spark.stop()
