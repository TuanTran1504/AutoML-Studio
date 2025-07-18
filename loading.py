import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier, XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.compose import ColumnTransformer
from collections import Counter
from sklearn.metrics import mean_absolute_error, make_scorer, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from tkinter import filedialog, ttk, messagebox

def pre_loading(train_path="train.csv", test_path=None):
    try:
        # Load train
        train = pd.read_csv(train_path)
        print("\nChecking for missing values in training data:")

        # Get missing columns for train
        train_missing = train.isnull().sum()
        train_missing = train_missing[train_missing > 0]

        # Build detailed missing info for train
        train_info = []
        for col in train_missing.index:
            dtype = train[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                col_type = "Numerical"
            elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                col_type = "Categorical"
            else:
                col_type = "Other"
            train_info.append({
                "Column": col,
                "Missing Count": train_missing[col],
                "Type": col_type
            })
        train_missing_df = pd.DataFrame(train_info)

        print(train_missing_df)

        # Load test if available
        if test_path:
            test = pd.read_csv(test_path)
            print("\nChecking for missing values in testing data:")

            test_missing = test.isnull().sum()
            test_missing = test_missing[test_missing > 0]

            test_info = []
            for col in test_missing.index:
                dtype = test[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    col_type = "Numerical"
                elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                    col_type = "Categorical"
                else:
                    col_type = "Other"
                test_info.append({
                    "Column": col,
                    "Missing Count": test_missing[col],
                    "Type": col_type
                })
            test_missing_df = pd.DataFrame(test_info)

            print(test_missing_df)
        else:
            test = None
            test_missing_df = pd.DataFrame()  # Empty if no test data

        return train, test, train_missing_df, test_missing_df

    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None, None, None
        
def loading(train_path="train.csv", test_path=None, num_method=None, cat_method=None, target_column=None, time_series=False, time_column=None, special_columns=None):
    try:
        # Load train bắt buộc
        train = pd.read_csv(train_path)

        # Load test nếu có
        test = pd.read_csv(test_path) if test_path else None
        print(train[special_columns].isna().sum())
        print("Train data loaded successfully!")
        
        if test is not None:
            print("Test data loaded successfully!")
        if special_columns is not None:
            for col in special_columns:
                mask = train[col].notna().cumsum() > 0
                train.loc[mask, col] = train.loc[mask, col].ffill()
        missing_ratio = train.isna().mean().sort_values(ascending=False)
        train = train.drop(columns=missing_ratio[missing_ratio > 0.5].index)
        train = train.dropna(subset=[target_column])
        # Xử lý missing data
        columns_to_fill = [col for col in train.columns if col != target_column]
        numeric_cols = train[columns_to_fill].select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = train[columns_to_fill].select_dtypes(include=["object", "category"]).columns.tolist()
        
        
        def fill_numeric(df, method):
            if df.empty or method == "None":
                return df
            if method == "mean":
                return df.fillna(df.mean(numeric_only=True))
            elif method == "median":
                return df.fillna(df.median(numeric_only=True))
            elif method == "zero":
                return df.fillna(0)
            elif method == "drop":
                return df.dropna()
            else:
                print("Unknown numeric fill method:", method)
                return df

        # ---------- Process Categorical Columns ----------
        def fill_categorical(df, method):
            # Check if DataFrame is empty
            if df.empty or method =="None":
                return df

            if method == "mode":
                mode_df = df.mode()
                if not mode_df.empty:
                    # Fill NaN values in each column with its mode
                    df = df.fillna(mode_df.iloc[0])
                else:
                    print("Warning: Mode could not be computed. No changes made.")
                return df

            elif method == "missing":
                return df.fillna("Missing")

            elif method == "drop":
                return df.dropna()
                
            else:
                print("Unknown categorical fill method:", method)
                return df
        
        # Xử lý train
        if num_method != "drop":
            train[numeric_cols] = fill_numeric(train[numeric_cols], num_method)
        if cat_method != "drop":
            train[categorical_cols] = fill_categorical(train[categorical_cols], cat_method)
        if num_method == "drop" or cat_method == "drop":
            before = len(train)
            dropped_train = train.dropna()
            after = len(dropped_train)
            dropped_percent = (before - after) / before * 100
            # If too much data is lost (e.g., over 10%), ask the user
            if dropped_percent > 10:
                response = messagebox.askyesno(
                        "Warning: Too Much Data Loss", "Do you want to proceed?"
                        )
                if response:
                    return dropped_train, None
                else:
                    messagebox.showinfo("Cancelled", "Please choose a different missing value method.")
                    return None, None  # Return None to indicate failure

            
        
        

        # Xử lý test nếu có
        if test is not None:
            if num_method != "drop":
                test[numeric_cols] = fill_numeric(test[numeric_cols], num_method)
            if cat_method != "drop":
                test[categorical_cols] = fill_categorical(test[categorical_cols], cat_method)
            if num_method == "drop" or cat_method == "drop":
                test = test.dropna()

        print("Missing data handled using:", num_method, cat_method)

        if test is not None:
            return train, test
        else:
            return train, None  # Để đồng nhất khi gọi hàm

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def model(cat_cols=None, num_cols=None, n_iter=20, cv=10):    
    if cat_cols and num_cols:
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ])
    else:
        preprocessor = 'passthrough'
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', XGBClassifier(random_state=8, eval_metric='auc'))
    ])
    
    search_space= {
        'clf__max_depth': Integer(2,8),
        'clf__learning_rate':Real(0.001, 1.0, prior='log-uniform'),
        'clf__subsample': Real(0.5, 1.0),
        'clf__colsample_bytree': Real(0.5, 1.0),
        'clf__colsample_bylevel': Real(0.5, 1.0),
        'clf__colsample_bynode': Real(0.5, 1.0),
        'clf__reg_lambda': Real(0.0, 10.0),
        'clf__reg_alpha': Real(0.0, 10.0),
        'clf__gamma': Real(0.0, 10.0)
    }
    opt = BayesSearchCV(pipe, search_space, cv=cv, n_iter=n_iter, scoring='roc_auc', random_state=8, verbose=True)
    return opt
def multiclass_model(cat_cols=None, rare_class_strategy="drop", min_class_size=5, merge_class_value=0, target_column=None,train=None, num_cols=None, n_iter=20, cv=10):
    if train is not None:
        class_counts = Counter(train[target_column])
        
        rare_classes = [cls for cls, count in class_counts.items() if count < min_class_size]
        if rare_classes:
                print(f"Rare classes detected: {rare_classes}")
                if rare_class_strategy == "drop":
                    train = train[~train[target_column].isin(rare_classes)]
                    print(f"Dropped rare classes. New class distribution: {Counter(train[target_column])}")

                elif rare_class_strategy == "merge":
                    train[target_column] = train[target_column].apply(
                        lambda x: merge_class_value if x in rare_classes else x
                    )
                    print(f"Merged rare classes into {merge_class_value}. New class distribution: {Counter(train[target_column])}")

                elif rare_class_strategy == "none":
                    print("Rare classes are kept as-is. Be aware this might cause errors during stratified splitting.")
                else:
                    print(f"Unknown rare_class_strategy: {rare_class_strategy}")
                    # ---------- Process Numerical Columns ----------
    df = train.drop(columns=target_column)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Preprocessing
    if cat_cols and num_cols:
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ])
    else:
        preprocessor = 'passthrough'

    # Pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', XGBClassifier(random_state=8, eval_metric='mlogloss'))
    ])

    # Hyperparameter space
    search_space = {
        'clf__max_depth': Integer(2, 8),
        'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'clf__subsample': Real(0.5, 1.0),
        'clf__colsample_bytree': Real(0.5, 1.0),
        'clf__reg_lambda': Real(0.0, 10.0),
        'clf__reg_alpha': Real(0.0, 10.0),
        'clf__gamma': Real(0.0, 10.0)
    }
    scorer = make_scorer(accuracy_score, greater_is_better=True)
    # Optimize Accuracy for multi-class 
    opt = BayesSearchCV(pipe, search_space, cv=cv, n_iter=n_iter, scoring=scorer, random_state=8, verbose=True)
    return opt, train
def regression_model(cat_cols=None, time_series=False, num_cols=None, log_transform=False, n_iter=20, cv=10):
    # Choose CV strategy
    cv = cv if not time_series else TimeSeriesSplit(n_splits=cv)

    # Preprocessing
    if cat_cols or num_cols:
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ])
    else:
        preprocessor = 'passthrough'

    # Base regressor pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=8, eval_metric='mae'))
    ])

    # Optional: Wrap with log-transform for the target
    if log_transform:
        pipe = TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=np.expm1
        )

    # Hyperparameter space
    search_space = {
        'regressor__regressor__max_depth': Integer(2, 8),
        'regressor__regressor__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'regressor__regressor__subsample': Real(0.5, 1.0),
        'regressor__regressor__colsample_bytree': Real(0.5, 1.0),
        'regressor__regressor__reg_lambda': Real(0.0, 10.0),
        'regressor__regressor__reg_alpha': Real(0.0, 10.0),
        'regressor__regressor__gamma': Real(0.0, 10.0)
    } if log_transform else {
        'regressor__max_depth': Integer(2, 8),
        'regressor__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'regressor__subsample': Real(0.5, 1.0),
        'regressor__colsample_bytree': Real(0.5, 1.0),
        'regressor__reg_lambda': Real(0.0, 10.0),
        'regressor__reg_alpha': Real(0.0, 10.0),
        'regressor__gamma': Real(0.0, 10.0)
    }

    # Use MAE as a scoring metric
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Bayesian Optimization
    opt = BayesSearchCV(
        pipe,
        search_space,
        cv=cv,
        n_iter=n_iter,
        scoring=scorer,
        random_state=8,
        verbose=True
    )

    return opt


    

def detect_class_type(y, class_threshold=15):
    num_classes = y.nunique()
    print(f"Detected {num_classes} unique classes in target variable.")
    print(f"Number of unique classes: {num_classes}")
    dtype = y.dtype
    y_min, y_max = y.min(), y.max()
    
    # First: Unique value check for binary
    if num_classes == 2:
        print(f"Binary classification detected. Classes: {list(y.unique())}, Unique values: {num_classes}")
        return "binary"
    if dtype == "object" or dtype == "category":
        # Categorical data with more than 2 unique values
        if num_classes > 2:
            print(f"Multi-class classification detected. Classes: {list(y.unique())}, Unique values: {num_classes}")
            return "multiclass"
        else:
            print(f"Single class detected in categorical data: {list(y.unique())}, Unique values: {num_classes}")
            return "unknown"

    if num_classes <= class_threshold and dtype.kind in 'biu':
        # Small number of unique integer classes → likely classification
        return "multiclass"
    elif dtype.kind in 'fc':
        return "regression" if num_classes > class_threshold else "multiclass"
    elif dtype.kind in 'biu' and (y_max - y_min) > class_threshold:
        return "regression"
    else:
        return "unknown"

