# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction
# MAGIC The notebook demonstrates the importance of feature engineering with domain knowledge with a 'controlled' experiment. It shows that relevant features can dramatically improve ML model performance. 
# MAGIC
# MAGIC
# MAGIC The notebook has two main components: 
# MAGIC 1. Input and output experiment data generation, 
# MAGIC 2. Model comparison with raw data, 'wrong' features, and features based on domain knowledge
# MAGIC
# MAGIC **Experiment Background**
# MAGIC
# MAGIC We have daily transactional prescription claims data for dozens patients with chronic illness for one year. Following clinical knowledge, we artifically generated "hospital visit" outcomes based on the medication adherence rate; that is, the patient will have a "hospital visit" if the adherence rate (measured by Proportion of Days Covered) is less than 80% in a year. 
# MAGIC
# MAGIC **Reference**
# MAGIC
# MAGIC Proportion of Days Covered (PDC) is the preferred method to measure medication adherence. The PDC threshold is the level above which the medication has a reasonable likelihood of achieving the most clinical benefit. Clinical evidence provides support for a standard PDC threshold of 80%. 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Preparation: A Simple Classification Model

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def simple_classification_model(X, y):
    """A simple classification model for demo purpose"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=16
    )

    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    return classification_report(y_test, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment Data Generation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Input Data: Randomly Generated Prescription Claims Data

# COMMAND ----------

import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducibility

num_of_days = 28 * 12
dates = pd.date_range(start="2023-01-01", periods=num_of_days, freq="D")
patient_ids = np.random.randint(1, 28 * 2, size=num_of_days)
covered_days = np.random.randint(28, 28 * 2, size=num_of_days)

df_rx_claims = pd.DataFrame(
    {"Date": dates, "Patient": patient_ids, "Covered Days": covered_days}
)
df_rx_claims.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Data: Outcome based on Clinical Knowledge

# COMMAND ----------

df_patient = df_rx_claims.groupby("Patient").agg(
    total_covered_days=("Covered Days", "sum"),
    avg_covered_days=("Covered Days", "mean"),
).reset_index()

df_patient = df_patient.rename(columns={"total_covered_days": "Total Covered Days", "avg_covered_days": "Avg. Covered Days"})

df_patient["Proportion of Days Covered (PDC)"] = (
    df_patient["Total Covered Days"] / num_of_days
)
df_patient["Hospital Visit"] = df_patient["Proportion of Days Covered (PDC)"] < 0.8
df_patient["Hospital Visit"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Performance Comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw Data

# COMMAND ----------

# Raw Data
df_rx_claims_with_outcome = df_rx_claims.join(df_patient[['Patient', 'Hospital Visit']].set_index('Patient'), on='Patient')
X = df_rx_claims_with_outcome[['Covered Days']]
y = df_rx_claims_with_outcome['Hospital Visit'] 
print(simple_classification_model(X, y))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering without Domain Knowledge

# COMMAND ----------

# Feature Engineering without Domain Knowledge
X = df_patient[['Avg. Covered Days']] 
y = df_patient['Hospital Visit'] 
print(simple_classification_model(X, y))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering with Domain Knowledge

# COMMAND ----------

# Feature Engineering with Domain Knowledge
X = df_patient[['Proportion of Days Covered (PDC)']] 
y = df_patient['Hospital Visit'] 
print(simple_classification_model(X, y))
