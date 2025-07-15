import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_curve, recall_score, precision_score, f1_score
from loading import loading, model, detect_class_type, multiclass_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train, test = loading("fetal_health.csv", test_path=None, target_column="fetal_health")


# X = train.drop(columns="Drug")
# y = train["Drug"]
# classes=detect_class_type(y)
# print(classes)
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=  y_encoded, random_state=8)
# cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
# opt=multiclass_model(cat_cols=cat_cols)
# opt.fit(X_train, y_train)
# print(opt.best_estimator_)
# print(opt.best_score_)