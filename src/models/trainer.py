# src/models/trainer.py
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
import config

class ModelManager:
    def __init__(self, model_dir=config.MODEL_SAVED_DIR, model_name="best_model.joblib"):
        """
        Manages cross-validation selection, training, and caching of ML models.
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, model_name)

    def get_candidate_models(self):
        return {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "Naive Bayes": GaussianNB(),
            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        }

    def train_and_select_best_model(self, X_train, X_test, y_train, y_test, scoring="roc_auc", cv_splits=5, force_retrain=False):
        """
        Checks if a saved model exists. If found, skips cross-validation & training.
        Otherwise, runs TimeSeriesSplit CV, selects the best model, fits, and serializes it.
        """
        if os.path.exists(self.model_path) and not force_retrain:
            print(f"--> Found existing saved model at: {self.model_path}")
            print("--> Skipping cross-validation and training.")
            loaded_model = self.load_model()
            model_type_name = type(loaded_model).__name__
            return self.model_path, model_type_name

        print(f"\nNo cached model found at '{self.model_path}' (or force_retrain=True). Starting training...")
        candidates = self.get_candidate_models()
        
        # Enforce expanding-window TimeSeriesSplit to prevent temporal leakage
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        print(f"=== Evaluating Algorithms via TimeSeriesSplit CV on Training Set ({scoring.upper()}) ===")
        
        best_name = None
        best_cv_score = -1.0
        best_model_obj = None

        for name, clf in candidates.items():
            scores = cross_val_score(clf, X_train, y_train, cv=tscv, scoring=scoring)
            mean_score = scores.mean()
            std_score = scores.std()
            
            print(f"-> {name:<22}: Mean {scoring.upper()} = {mean_score:.4f} (+/- {std_score:.4f})")
            
            if mean_score > best_cv_score:
                best_cv_score = mean_score
                best_name = name
                best_model_obj = clf

        print(f"\nWinner Algorithm: '{best_name}' (CV Score: {best_cv_score:.4f})")
        
        best_model_obj.fit(X_train, y_train)
        
        test_preds = best_model_obj.predict(X_test)
        test_probs = best_model_obj.predict_proba(X_test)[:, 1] if hasattr(best_model_obj, "predict_proba") else test_preds
        
        print(f"\n=== Final Holdout Test Metrics ({best_name}) ===")
        print(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}")
        print(f"AUC:      {roc_auc_score(y_test, test_probs):.4f}\n")
        
        joblib.dump(best_model_obj, self.model_path)
        print(f"Model successfully saved to: {self.model_path}")
        
        return self.model_path, best_name

    def load_model(self):
        return joblib.load(self.model_path)