#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Load & Preprocessing (jalan waktu diimport) ───────────────────────────

df = pd.read_csv("../dataset/loan_sanction_train.csv")

# Penanganan Missing Value
median_lount_amount = df['LoanAmount'].median()
df['LoanAmount'] = df['LoanAmount'].fillna(median_lount_amount)

mode_term = df['Loan_Amount_Term'].mode()[0]
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(mode_term)

mode_credit = df["Credit_History"].mode()[0]
df['Credit_History'] = df['Credit_History'].fillna(mode_credit)

mode_gender = df["Gender"].mode()[0]
df['Gender'] = df['Gender'].fillna(mode_gender)

mode_married = df["Married"].mode()[0]
df['Married'] = df['Married'].fillna(mode_married)

df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
modus_dependents = df['Dependents'].mode()[0]
df['Dependents'] = df['Dependents'].fillna(modus_dependents)

df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

# Penanganan Outliers - Drop
df_temp = df.copy()
cols_to_ignore = ['Dependents', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

while True:
    num_cols = df_temp.select_dtypes(include=['int64', 'float64'])
    num_cols = num_cols.drop(columns=cols_to_ignore, errors='ignore')

    Q1 = num_cols.quantile(0.25)
    Q3 = num_cols.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((num_cols < (Q1 - 1.5 * IQR)) | (num_cols > (Q3 + 1.5 * IQR)))
    rows_with_outliers = outliers.any(axis=1)

    if rows_with_outliers.sum() == 0:
        break

    df_temp = df_temp[~rows_with_outliers]

df_clean_outliers = df_temp.reset_index(drop=True)

# Penanganan Outliers - Replace dengan median
df_outliers_replace = df.copy()
cols = ['ApplicantIncome', 'CoapplicantIncome']

while True:
    num_cols = df_outliers_replace[cols]

    Q1 = num_cols.quantile(0.25)
    Q3 = num_cols.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((num_cols < lower) | (num_cols > upper))

    if outliers.sum().sum() == 0:
        break

    for col in cols:
        median = df_outliers_replace[col].median()
        df_outliers_replace[col] = df_outliers_replace[col].mask(
            (df_outliers_replace[col] < lower[col]) |
            (df_outliers_replace[col] > upper[col]),
            median
        )


# ─── Hanya jalan kalau dirun langsung, TIDAK waktu diimport ─────────────────

if __name__ == "__main__":
    print(os.getcwd())
    print(df.head())

    print("\nDeskripsi statistik")
    print(df.describe())

    print(df['Dependents'].unique())
    df.info()
    print(df.isnull().sum())

    # Boxplot awal
    num_cols_display = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=num_cols_display)
    plt.xticks(rotation=45)
    plt.show()

    # Outlier count awal
    Q1 = num_cols_display.quantile(0.25)
    Q3 = num_cols_display.quantile(0.75)
    IQR = Q3 - Q1
    outliers_check = ((num_cols_display < (Q1 - 1.5 * IQR)) | (num_cols_display > (Q3 + 1.5 * IQR)))
    print(outliers_check.sum())

    print(f"Modus loan_term : {mode_term}")
    print(f"Modus credit history : {mode_credit}")
    print(f"Modus Gender : {mode_gender}")
    print(f"Modus Married : {mode_married}")
    print(f"Modus Dependents : {modus_dependents} dengan type : {type(modus_dependents)}")

    print(df.isnull().sum())

    # Boxplot setelah missing value
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=num_cols_display)
    plt.xticks(rotation=45)
    plt.show()

    # Outlier count setelah missing value
    num_cols2 = df.select_dtypes(include=['int64', 'float64'])
    Q1 = num_cols2.quantile(0.25)
    Q3 = num_cols2.quantile(0.75)
    IQR = Q3 - Q1
    outliers2 = ((num_cols2 < (Q1 - 1.5 * IQR)) | (num_cols2 > (Q3 + 1.5 * IQR)))
    print(outliers2.sum())

    # Boxplot df_clean_outliers
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_clean_outliers)
    plt.xticks(rotation=45)
    plt.show()

    df_clean_outliers.info()

    # Before vs After plot outliers replace
    for col in cols:
        plt.figure()

        plt.subplot(1, 2, 1)
        sns.boxplot(data=df[[col]])
        plt.title(f'Before - {col}')

        plt.subplot(1, 2, 2)
        sns.boxplot(data=df_outliers_replace[[col]])
        plt.title(f'After - {col}')

        plt.tight_layout()
        plt.show()
