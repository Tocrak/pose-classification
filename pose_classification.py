import argparse
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


def get_keypoints_data(image_dirs):
    data = []
    examples_count = []
    for directory in image_dirs:
        op_model.run(image_dir=directory)
        keypoints_data = flatten_keypoints_data(op_model.keypoints)
        data.extend(keypoints_data)
        examples_count.append(len(keypoints_data))
    return {"data": data, "examples_count": examples_count}


def train_test_models(models):
    for model in models:
        print("Current Model:", type(model).__name__)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        # Add  average="micro"/"macro"/"weighted"  to Recall, Precision and F-Score for multiclass examples
        print("Recall:", recall_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("F-score:", f1_score(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), "\n")


# Give the image directories as command line arguments for the script
# Each directory will be considered a different class of items
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-d", "--image-dirs", nargs='+', type=str, required=True,
                        help="Provide multiple image directories")
args = arg_parser.parse_args()

op_model = OpenposeRunner()
# Decrease the net resolution (if not enough VRAM). May affect models results.
# op_model.params["net_resolution"] = "400x320"

keypoints_data = get_keypoints_data(args.image_dirs)

X_data = keypoints_data["data"]
# Uses list comprehension to give numerical labels to each item (containing keypoint data)
# representing the classes (e.g. 0/1 for static/dynamic poses)
y_data = [class_index for index, data in enumerate(keypoints_data["examples_count"]) for class_index in [index] * data]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=1)

# Test 5 machine learning classification models
models = [
    # Add  multi_class='multinomial'  parameter to LogisticRegression for multiclass examples
    LogisticRegression(max_iter=50000),
    SVC(kernel='rbf'),
    MultinomialNB(),
    RandomForestClassifier(random_state=1),
    KNeighborsClassifier(weights="uniform"),
]

train_test_models(models)
