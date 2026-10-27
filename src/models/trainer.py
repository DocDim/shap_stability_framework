# src/models/trainer.py
import os
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import config

log = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_dir=config.MODEL_SAVED_DIR):
        """
        Trains and caches every candidate model (no selection step).
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def get_candidate_models(self):
        return {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=config.SEED, class_weight="balanced", n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=config.SEED),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.SEED, class_weight="balanced"),
            "Naive Bayes": GaussianNB(),
            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=config.SEED)
        }

    @staticmethod
    def _matches_features(clf, X):
        """A cached model is reusable only if it was fitted on exactly these columns, in this order."""
        names = getattr(clf, "feature_names_in_", None)
        return names is not None and list(names) == list(X.columns)

    def train_all_models(self, X_train, X_test, y_train, y_test, domain_tag, force_retrain=False):
        """
        Fits every candidate model and caches each one as '<domain_tag>_<model_slug>.joblib'.
        Cached models fitted on different columns are retrained. Returns {name: (model, holdout_auc)}.
        """
        fitted = {}
        for name, clf in self.get_candidate_models().items():
            slug = name.lower().replace(" ", "_")
            path = os.path.join(self.model_dir, f"{domain_tag}_{slug}.joblib")

            cached = joblib.load(path) if os.path.exists(path) and not force_retrain else None
            if cached is not None and self._matches_features(cached, X_train):
                clf = cached
                log.info("--> Loaded cached '%s' from: %s", name, path)
            else:
                if cached is not None:
                    log.warning("--> Cached '%s' was fitted on different features; retraining", name)
                log.info("--> Training '%s'...", name)
                clf.fit(X_train, y_train)
                joblib.dump(clf, path)

            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            log.info("    %-22s holdout AUC = %.4f", name, auc)
            if auc < config.MIN_MODEL_AUC:
                log.warning("    %s AUC %.4f is below %.2f; its explanations describe a near-random model",
                            name, auc, config.MIN_MODEL_AUC)
            fitted[name] = (clf, auc)
        return fitted
