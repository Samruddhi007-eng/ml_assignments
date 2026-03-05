import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# Load Dataset
def load_data(path, target):
    df = pd.read_csv(path)
    df[target] = df[target].map({'ham': 0, 'spam': 1})
    return df


# TF-IDF Preprocessing
def preprocess(messages):

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_df=0.9,
        min_df=2
    )

    X = tfidf.fit_transform(messages)
    return X, tfidf


# Define Models
def get_models():

    nb = MultinomialNB()

    lr = LogisticRegression(max_iter=1000)

    svm = CalibratedClassifierCV(
        LinearSVC(),
        method="sigmoid"
    )

    hard_vote = VotingClassifier(
        estimators=[('nb', nb), ('lr', lr), ('svm', svm)],
        voting='hard'
    )

    soft_vote = VotingClassifier(
        estimators=[('nb', nb), ('lr', lr), ('svm', svm)],
        voting='soft'
    )

    stacking = StackingClassifier(
        estimators=[('nb', nb), ('lr', lr), ('svm', svm)],
        final_estimator=LogisticRegression()
    )

    stump = DecisionTreeClassifier(max_depth=1)

    adaboost = AdaBoostClassifier(
        estimator=stump,
        n_estimators=100,
        random_state=42
    )

    return {
        "NaiveBayes": nb,
        "LogisticRegression": lr,
        "LinearSVM": svm,
        "HardVoting": hard_vote,
        "SoftVoting": soft_vote,
        "Stacking": stacking,
        "AdaBoost_Stumps": adaboost
    }


# Evaluation using K-Fold
def evaluate_models(X, y, models, k=5):

    skf = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=42
    )

    results = []

    for name, model in models.items():

        precision, recall, f1, roc = [], [], [], []

        print(f"\nTraining: {name}")

        for train_idx, test_idx in skf.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = preds

            precision.append(precision_score(y_test, preds))
            recall.append(recall_score(y_test, preds))
            f1.append(f1_score(y_test, preds))
            roc.append(roc_auc_score(y_test, probs))

        results.append([
            name,
            np.mean(precision), np.std(precision),
            np.mean(recall), np.std(recall),
            np.mean(f1), np.std(f1),
            np.mean(roc), np.std(roc)
        ])

    columns = [
        "Model",
        "Precision_mean", "Precision_std",
        "Recall_mean", "Recall_std",
        "F1_mean", "F1_std",
        "ROC_mean", "ROC_std"
    ]

    return pd.DataFrame(results, columns=columns)


# Final Prediction File
def generate_predictions(model, X, y):

    model.fit(X, y)

    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = preds

    output = pd.DataFrame({
        "MessageId": range(len(y)),
        "Actual": y,
        "Predicted": preds,
        "Probability": probs
    })

    output.to_csv("final_model_predictions.csv", index=False)


# Main Function
def main(data_path, target, k):

    df = load_data(data_path, target)

    X, vectorizer = preprocess(df["message"])
    y = df[target]

    models = get_models()

    results = evaluate_models(X, y, models, k)

    results.to_csv("ensemble_comparison.csv", index=False)

    print("\nFinal Results:")
    print(results)

    # Use stacking as final recommended model
    generate_predictions(models["Stacking"], X, y)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="label")
    parser.add_argument("--kfold", type=int, default=5)

    args = parser.parse_args()

    main(args.data, args.target, args.kfold)