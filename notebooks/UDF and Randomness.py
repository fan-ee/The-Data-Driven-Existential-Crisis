# Databricks notebook source
data = [['Mavs'], 
        ['Nets'], 
        ['Lakers'], 
        ['Kings'], 
        ['Hawks'],
        ['Wizards'],
        ['Magic'],
        ['Jazz'],
        ['Thunder'],
        ['Spurs']]
  
columns = ['experiment'] 
  
df_experiment = spark.createDataFrame(data, columns)   

# COMMAND ----------

import random
import pyspark.sql.functions as F
from pyspark.sql.types import LongType


@F.udf(LongType())
def udf_rand():
    return random.randint(1, 9)


@F.udf(LongType())
def udf_double(x):
    return 2 * x


df_rand = df_experiment.withColumn("rand", udf_rand())
df_udf_double = df_rand.withColumn("udf double", udf_double(F.col("rand")))
df_built_in_double = df_rand.withColumn("double", 2 * F.col("rand"))
df_udf_double.explain()
df_built_in_double.explain()

# COMMAND ----------

import random
import pyspark.sql.functions as F
from pyspark.sql.types import LongType


@F.udf(LongType())
def udf_rand():
    return random.randint(1, 9)


@F.udf(LongType())
def udf_double(x):
    return 2 * x


df_rand = df_experiment.withColumn("rand", udf_rand())
df_built_in_double = df_rand.withColumn("double", 2 * F.col("rand"))
df_built_in_double.explain()
