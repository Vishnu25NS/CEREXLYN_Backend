import numpy as np
from sklearn.ensemble import VotingClassifier

def fix_proba(clf, X, all_classes):
    if not hasattr(clf, "predict_proba"):
        preds = clf.predict(X)
        proba = np.zeros((len(preds), len(all_classes)))
        for i, p in enumerate(preds):
            proba[i, all_classes.index(p)] = 1.0
        return proba

    proba = clf.predict_proba(X)
    n_samples = proba.shape[0]
    fixed = np.zeros((n_samples, len(all_classes)), dtype=float)

    for i, cls in enumerate(clf.classes_):
        if cls in all_classes:
            j = all_classes.index(cls)
            fixed[:, j] = proba[:, i]
    return fixed


class FixedVotingClassifier(VotingClassifier):
    def __init__(self, *args, all_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        if all_classes is None:
            raise ValueError("FixedVotingClassifier requires all_classes parameter.")
        self.all_classes = list(all_classes)

    def predict_proba(self, X):
        probas = []
        for name, est in self.estimators_:
            prob = fix_proba(est, X, self.all_classes)
            probas.append(prob)

        stacked = np.stack(probas, axis=0)

        if self.weights is not None:
            weights = np.array(self.weights)[:, None, None]
            avg = np.sum(stacked * weights, axis=0) / np.sum(self.weights)
        else:
            avg = np.mean(stacked, axis=0)

        return avg

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
