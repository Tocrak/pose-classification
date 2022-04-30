from itertools import chain
from openpose_runner import OpenposeRunner
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def flatten_keypoints_data(keypoints):
    data = []
    for element in keypoints:
        data.append(list(chain(*element)))
    return data


def train_test_models(models):
    for model in models:
        print("Current Model:", type(model).__name__)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))


op_model = OpenposeRunner()
# Decrease the net resolution (if not enough VRAM)
# op_model.params["net_resolution"] = "400x320"

op_model.run(image_dir="./media/static")
keypoints_static = flatten_keypoints_data(op_model.keypoints)
op_model.run(image_dir="./media/dynamic")
keypoints_dynamic = flatten_keypoints_data(op_model.keypoints)

X_data = keypoints_static + keypoints_dynamic
y_data = [0] * len(keypoints_static) + [1] * len(keypoints_dynamic)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=1)

# Test 5 machine learning classification models
models = [
    LogisticRegression(max_iter=10000),
    SVC(kernel='rbf'),
    MultinomialNB(),
    RandomForestClassifier(random_state=1),
    KNeighborsClassifier(weights="uniform"),
]

train_test_models(models)
