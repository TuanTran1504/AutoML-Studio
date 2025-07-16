import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from loading import loading, model, detect_class_type, multiclass_model, pre_loading, regression_model
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_curve, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Biến toàn cục để lưu data
global_train = None
global_test = None
global_target = None
global_train_path=None
global_test_path=None
global_missing = "No Missing Data"
global_problem_types = None
global_ts_mode = False

def chon_file():
    global global_train_path, global_test_path

    global_train_path = None
    global_test_path = None

    filepath_train = filedialog.askopenfilename(title="Chọn file train", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if not filepath_train:
        print("Chưa chọn file train.")
        return
    filepath_test = filedialog.askopenfilename(title="Chọn file test", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if not filepath_test:
        print("Khong chon file test")
    global_train_path=filepath_train
    global_test_path=filepath_test if filepath_test else None

    pre_data_train, pre_data_test, train_na, test_na = pre_loading(global_train_path, global_test_path)
    if train_na.empty:
        global_missing = "No Missing Data."
        label.config(text=f" Do not need to handle missing data. Proceed to choose target.")
    else: 
        num_missing = train_na[train_na["Type"] == "Numerical"]
        cat_missing = train_na[train_na["Type"] == "Categorical"]
        if not num_missing.empty and not cat_missing.empty:
            global_missing = "Missing Numerical and Categorical"
        elif not num_missing.empty:
            global_missing = "Missing Numerical"
        elif not cat_missing.empty:
            global_missing = "Missing Categorical"
                
        messagebox.showinfo("Missing Data Info", global_missing)
        label.config(text=f" Choose method(s) to handle'{global_missing}' ")
    

 
    columns = pre_data_train.columns.tolist()

        # Hiển thị column vào combobox
    target_combobox['values'] = columns
    time_col_combobox['values'] = columns
    special_listbox.delete(0, tk.END)
    for col in columns:
        special_listbox.insert(tk.END, col)
    target_combobox.set('')  # reset combobox
    
    
def chon_target():
    global global_train, global_test, global_target
    target_column = target_column_var.get().strip()
    global_target=target_column
    num_fill_method = selected_num_method.get()
    cat_fill_method = selected_cat_method.get()
    if not target_column:
        label.config(text="Bạn chưa chọn target column.")
        return
    # Get selected special columns
    selected_indices = special_listbox.curselection()
    special_columns = [special_listbox.get(i) for i in selected_indices]

    # Handle special columns first
    
    try:
        # Load data và lưu vào biến global
        train, test = loading(global_train_path, global_test_path if global_test_path else None, num_fill_method, cat_fill_method, target_column=target_column, special_columns=special_columns)
        if target_column not in train.columns:
            label.config(text=f" Cột '{target_column}' không tồn tại trong train data.")
            return
        global_train = train
        global_test = test
        print(global_train[global_target].describe())
        if test is not None:
            label.config(text=f"Đã load train & test.\nTrain shape: {train.shape}\nTest shape: {test.shape}")
        else:
            label.config(text=f"Đã load train.\nTrain shape: {train.shape}\nTest file chưa được chọn.")
        
        print("File đã load xong. Xin hay chon problem.")
    except Exception as e:
        label.config(text=f"Lỗi khi load file: {e}")
        print(e)

def detect_class():
    global global_problem_types, global_target, global_train
    if global_target is None:
        label.config(text="Vui lòng chọn target column trước!")
        return
    print(len(global_train))
    pred_type = detect_class_type(global_train[global_target])
    if global_problem_types is None:
        global_problem_types = selected_prob.get()
    print(f"User selected: {global_problem_types}, Detected from data: {pred_type}")
    if global_problem_types == pred_type:
        label.config(text=f"Problem type '{global_problem_types}' đã được xác nhận.")

    else:
        label.config(text=f"Problem type '{global_problem_types}' không khớp với dữ liệu. Dữ liệu hiện tại là '{pred_type}'. Vui lòng chọn lại.")
        global_problem_types = None
        selected_prob.set('')
        return

def toggle_time_column_selection():
    global global_ts_mode
    if is_time_series.get():
        global_ts_mode = True
        time_col_label.pack(pady=2)
        time_col_combobox.pack(pady=5)
    else:
        global_ts_mode = False
        time_col_label.pack_forget()
        time_col_combobox.pack_forget()


def train_model():
    global global_train, global_test, global_target, global_problem_types

    if global_train is None or global_target is None:
        label.config(text="Vui lòng chọn file trước!")
        return

    try:
        X = global_train.drop(columns=global_target)
        y = global_train[global_target]
        if global_problem_types=="binary":
            print("Using binary classification model")
            unique_values = sorted(y.unique())
            #Chekc for binary classification
            if unique_values == [0, 1]:
                y_encoded = y
            else:
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            # Tách tập train/validation
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=8)

            # Khởi tạo và train model
            opt = model(cat_cols)
            opt.fit(X_train, y_train)

            # Hiển thị kết quả
            print("Best estimator:", opt.best_estimator_)
            print("Best training score:", opt.best_score_)
            label.config(text=f"Đã train model binary.\nBest estimator: {opt.best_estimator_}\nBest score: {opt.best_score_}")
        elif global_problem_types == "multiclass":
            print(global_train)
            cat_cols = global_train.select_dtypes(include=['object', 'category']).columns.tolist()
            opt, global_train=multiclass_model(cat_cols=cat_cols, target_column=global_target, train=global_train)
            print("Using multiclass model")
            X = global_train.drop(columns=global_target)
            y = global_train[global_target]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify= y_encoded, random_state=8)
            print("Available columns:", global_train.columns.tolist())
            print("Target column:", global_target)
            opt.fit(X_train, y_train)
            print(opt.best_estimator_)
            print(opt.best_score_)
            metric_name = opt.scoring._score_func.__name__

            label.config(text=f"Đã train model multiclass.\nBest estimator: {opt.best_estimator_}\n{metric_name}={opt.best_score_}")
        else:
            print("Using regression model")

            if is_time_series.get():
                time_col = time_col_combobox.get()
                if not time_col:
                    label.config(text="Vui lòng chọn cột thời gian cho time series!")
                    return
                global_train['target_lag_1'] = global_train[global_target].shift(1)
                global_train['target_lag_2'] = global_train[global_target].shift(2)
                global_train.dropna(inplace=True)
                global_train[time_col] = pd.to_datetime(global_train[time_col])
                global_train['month'] = global_train[time_col].dt.month
                global_train['weekday'] = global_train[time_col].dt.weekday
                global_train['year'] = global_train[time_col].dt.year
                global_train.sort_values(by=time_col, inplace=True)
                global_train.drop(columns=[time_col], inplace=True)
                split_index = int(len(global_train) * 0.8)
                train = global_train.iloc[:split_index]
                test = global_train.iloc[split_index:]

                X_train = train.drop(columns=[global_target])
                y_train = train[global_target]
                X_test = test.drop(columns=[global_target])
                y_test = test[global_target]
                cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

                opt=regression_model(cat_cols=cat_cols, time_series=global_ts_mode, num_cols=num_cols, log_transform=True)
                opt.fit(X_train, y_train)
            else:
                X = global_train.drop(columns=global_target)
                y = global_train[global_target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
                cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

                opt=regression_model(cat_cols=cat_cols, time_series=global_ts_mode, num_cols=num_cols, log_transform=True)
                opt.fit(X_train, y_train)
            print(opt.best_estimator_)
            print(opt.best_score_)
            y_pred = opt.predict(X_test)
            

            print("MAE:", mean_absolute_error(y_test, y_pred))
            print("MSE:", mean_squared_error(y_test, y_pred))
            label.config(text=f"Đã train model regression.\nBest estimator: {opt.best_estimator_}\nBest score: {opt.best_score_}")

    except Exception as e:
        label.config(text=f"Lỗi khi train: {e}")
        print(e)

root = tk.Tk()
root.title("Chọn file train và test")
root.geometry("1000x1000")

# Nút chọn file
btn = tk.Button(root, text="Chọn file", command=chon_file)
btn.pack(pady=10)

# Dropdown for numeric imputation
num_methods = ["None","mean", "median", "zero", "drop"]
selected_num_method = tk.StringVar(root)
selected_num_method.set(num_methods[0])
num_label = tk.Label(root, text="Numeric missing fill:")
num_label.pack()
num_dropdown = tk.OptionMenu(root, selected_num_method, *num_methods)
num_dropdown.pack()

# Dropdown for categorical imputation
cat_methods = ["None","mode", "missing", "drop"]
selected_cat_method = tk.StringVar(root)
selected_cat_method.set(cat_methods[0])
cat_label = tk.Label(root, text="Categorical missing fill:")
cat_label.pack()
cat_dropdown = tk.OptionMenu(root, selected_cat_method, *cat_methods)
cat_dropdown.pack()

# Cập nhật danh sách cột đặc biệt
special_label = tk.Label(root, text="Chọn các cột chỉ có giá trị ở một vài thời điểm:")
special_label.pack()

special_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, exportselection=0, height=6)
special_listbox.pack(pady=5)

# Label target
target_label = tk.Label(root, text="Chọn target column:")
target_label.pack()

# Combobox chọn target
target_column_var = tk.StringVar()
target_combobox = ttk.Combobox(root, textvariable=target_column_var, state="readonly")
target_combobox.pack(pady=5)

# Nút xác nhận target
btn_target = tk.Button(root, text="Confirm handle missing data", command=chon_target)
btn_target.pack(pady=10)

# Dropdown for problem type
problem_types = ["binary", "multiclass", "regression"]

selected_prob = tk.StringVar(root)
selected_prob.set(problem_types[0])
prob_label = tk.Label(root, text="Problem type:")
prob_label.pack()
prob_dropdown = tk.OptionMenu(root, selected_prob, *problem_types)
prob_dropdown.pack()

# Checkbox to indicate if it's a time series problem
is_time_series = tk.BooleanVar()
chk_time_series = tk.Checkbutton(root, text="This is a time series problem", variable=is_time_series, command=lambda: toggle_time_column_selection())
chk_time_series.pack(pady=5)

# Time column dropdown (initially hidden)
time_col_label = tk.Label(root, text="Chọn time column:")
time_col_combobox = ttk.Combobox(root, state="readonly")

# Nút bắt đầu train
btn_class = tk.Button(root, text="Confirm Preprocessing", command=detect_class)
btn_class.pack(pady=10)

# Nút bắt đầu train
btn_train = tk.Button(root, text="Bắt đầu train", command=train_model)
btn_train.pack(pady=10)

# Label trạng thái
label = tk.Label(root, text="Chưa chọn file nào")
label.pack(pady=10)

# Chạy giao diện
root.mainloop()