from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, mean_squared_error)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
import pickle
import json

from models import Dataset, ModelEntry

# ===== Model wrapper/helper =====
class ModelWrapper:
    """
    Thin wrapper to standardize training/predicting and DB serialization.
    model_type: one of ('linear_regression','logistic_regression','decision_tree',
                       'bagging','boosting','random_forest','svm','mlp')
    """
    def __init__(self, model_type, regression=True, model=None, params=None):
        self.model_type = model_type
        self.model = model
        self.params = params or {}
        self.metrics = {}
        self.regression = regression

    def train(self, X, y, progress_callback=None, pause_check=None, early_stop_check=None, start_epoch=0, keep_model=False, **train_kwargs):
        # X,y are pandas or numpy-like
        
        # For MLP: Create a fresh model instance unless keep_model=True (for early-stopped retraining)
        if self.model_type == 'mlp':
            if not keep_model:
                # Starting fresh - create new model
                self.model = self._instantiate_from_type(self.regression, self.model_type, self.params)
            # else: keep existing model from blob (for early-stopped retraining)
        elif self.model is None:
            self.model = self._instantiate_from_type(self.regression, self.model_type, self.params)

        # Track the final epoch reached for resuming later
        self.final_epoch = start_epoch

        # For MLP with streaming support
        if self.model_type == 'mlp' and progress_callback:
            print(f"[MLP Training] Starting epoch-wise training with progress callback (start_epoch={start_epoch})")

            # Get max iterations from params (NOT from model which may have stale value)
            max_iter = self.params.get('max_iter', 200)
            print(f"[MLP Training] Max iterations from params: {max_iter}")

            # If the model is a Pipeline (classification path), adjust the final estimator's params
            estimator = None
            is_pipeline = hasattr(self.model, 'named_steps') and 'estimator' in getattr(self.model, 'named_steps', {})
            try:
                if is_pipeline:
                    # Configure warm start and 1-iteration training per fit call on the estimator
                    self.model.set_params(estimator__warm_start=True)
                    # Disable early stopping to avoid validation-split delays and premature stop
                    self.model.set_params(estimator__early_stopping=False)
                    self.model.set_params(estimator__max_iter=1)
                    estimator = self.model.named_steps['estimator']
                    # Fit the preprocessor ONCE to avoid repeated expensive refits per epoch
                    preprocessor = self.model.named_steps.get('preprocess', None)
                    X_fit = X
                    if preprocessor is not None:
                        try:
                            preprocessor.fit(X, y)
                            X_fit = preprocessor.transform(X)
                        except Exception as pe:
                            print(f"[MLP Training] Preprocessor fit/transform failed, falling back to pipeline.fit each epoch: {pe}")
                            preprocessor = None
                    # Store prepared features for reuse in the loop
                    prepared = {'X': X_fit, 'pre': preprocessor}
                else:
                    # MLPRegressor path (no pipeline)
                    self.model.warm_start = True
                    # Disable early stopping (MLP defaults to False, but be explicit)
                    if hasattr(self.model, 'early_stopping'):
                        self.model.early_stopping = False
                    # Train one iteration per fit call
                    self.model.max_iter = 1
                    estimator = self.model
                    prepared = {'X': X}
            except Exception as e:
                print(f"[MLP Training] Warning: Failed to configure warm-start incremental training: {e}")
                estimator = self.model
                prepared = {'X': X}

            # Track previous loss for convergence
            prev_loss = float('inf')
            # Pull tolerance from the actual estimator if available
            try:
                tol = getattr(estimator, 'tol', 1e-4)
            except Exception:
                tol = 1e-4

            for epoch in range(start_epoch, max_iter):
                # Check if early stop was requested
                if early_stop_check and early_stop_check():
                    print(f"[MLP Training] Early stop requested - saving progress at epoch {self.final_epoch}")
                    break

                # Check if training is paused
                if pause_check and pause_check():
                    print(f"[MLP Training] Training paused at epoch {epoch + 1}")
                    import time
                    while pause_check():
                        if early_stop_check and early_stop_check():
                            print(f"[MLP Training] Early stop requested during pause - saving progress at epoch {self.final_epoch}")
                            break
                        time.sleep(0.5)
                    print(f"[MLP Training] Training resumed at epoch {epoch + 1}")

                # Train exactly one iteration per call (warm_start keeps weights)
                try:
                    if is_pipeline and prepared.get('pre') is not None:
                        # Train only the estimator with pre-transformed features
                        estimator.fit(prepared['X'], y)
                    else:
                        # Fallback: train the model (pipeline or raw estimator)
                        self.model.fit(X, y)
                except Exception as fit_err:
                    print(f"[MLP Training] Fit error at epoch {epoch + 1}: {fit_err}")
                    raise

                # Calculate metrics for this epoch
                if is_pipeline and prepared.get('pre') is not None:
                    train_preds = estimator.predict(prepared['X'])
                else:
                    train_preds = self.model.predict(X)
                # Get loss from the underlying estimator when using a Pipeline
                try:
                    current_estimator = estimator
                    # Refresh reference in case Pipeline cloned estimator internally
                    if is_pipeline and hasattr(self.model, 'named_steps') and 'estimator' in self.model.named_steps:
                        current_estimator = self.model.named_steps['estimator']
                    train_loss = getattr(current_estimator, 'loss_', 0.0)
                except Exception:
                    train_loss = 0.0

                if self.regression:
                    train_metric = mean_squared_error(y, train_preds)
                else:
                    train_metric = accuracy_score(y, train_preds)

                print(f"[MLP Training] Epoch {epoch + 1}/{max_iter} - Loss: {float(train_loss):.6f}, {'MSE' if self.regression else 'Accuracy'}: {train_metric:.4f}")

                # Send progress update with regression flag
                progress_callback({
                    'epoch': epoch + 1,
                    'loss': float(train_loss),
                    'metric': float(train_metric),
                    'regression': self.regression
                })

                # Update final epoch after successful completion
                self.final_epoch = epoch + 1

                # Check for convergence (loss not improving)
                if epoch > start_epoch and abs(prev_loss - float(train_loss)) < tol:
                    print(f"[MLP Training] Converged at epoch {epoch + 1}")
                    break

                prev_loss = float(train_loss)

            print(f"[MLP Training] Training completed at epoch {self.final_epoch}")
        else:
            # For scikit-learn style models (standard training)
            if hasattr(self.model, 'fit'):
                self.model.fit(X, y)
            else:
                # If the model is custom and doesn't have fit, raise
                raise ValueError("Model does not support .fit() - provide a fitted model blob for 'custom' models")

    def predict(self, X):
        if self.model is None:
            raise ValueError("No model loaded")
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            # If model is pickled custom object without predict, try calling it as function
            try:
                return self.model(X)
            except Exception as e:
                raise ValueError(f"Model has no predict method and is not callable: {e}")

    def evaluate(self, X, y):
        preds = self.predict(X)
        if self.regression:
            self.metrics['mse'] = float(mean_squared_error(y, preds))
        else:
            # classification metrics
            try:
                self.metrics['accuracy'] = float(accuracy_score(y, preds))
            except Exception:
                self.metrics['accuracy'] = None
        return self.metrics

    def to_db_record(self, name, dataset_id, user_id):
        model_blob = pickle.dumps(self.model)
        # convert params and metrics to json strings
        params = json.dumps(self.params)
        metrics = json.dumps(self.metrics)
        return ModelEntry(
            name=name,
            model_type=self.model_type,
            params=params,
            metrics=metrics,
            model_blob=model_blob,
            dataset_id=dataset_id,
            user_id=user_id
        )

    @staticmethod
    def from_db_record(record: ModelEntry):
        # convert params and metrics from json strings to dicts
        params = json.loads(record.params) if record.params else {}
        metrics = json.loads(record.metrics) if record.metrics else {}
        model = pickle.loads(record.model_blob)
        dataset = Dataset.query.get(record.dataset_id)
        regression = dataset.regression if dataset else False
        wrapper = ModelWrapper(model_type=record.model_type, model=model, params=params, regression=regression)
        wrapper.metrics = metrics
        return wrapper

    @staticmethod
    def _instantiate_from_type(regression, model_type, params):
        # Map types to sklearn classes
        if model_type == 'linear_regression':
            if not regression:
                raise ValueError("Linear Regression model requires regression=True")
            # Add imputers and encoding for categorical
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            base = LinearRegression(**(params or {}))
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'logistic_regression':
            if regression:
                raise ValueError("Logistic Regression model requires regression=False")
            base = LogisticRegression(**(params or {}))
            pre = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))],
                remainder='passthrough'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'decision_tree':
            if regression:
                # Add imputers and encoding for categorical
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = DecisionTreeRegressor(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = DecisionTreeClassifier(**(params or {}))
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'random_forest':
            if regression:
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = RandomForestRegressor(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = RandomForestClassifier(**(params or {}))
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'bagging':
            if regression:
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = BaggingRegressor(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = BaggingClassifier(**(params or {}))
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'boosting':
            if regression:
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = GradientBoostingRegressor(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = AdaBoostClassifier(**(params or {}))
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'svm':
            if regression:
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = SVR(**(params or {}))
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            # Ensure we don't pass duplicate 'probability' kwarg
            _params = dict(params or {})
            if 'probability' not in _params:
                _params['probability'] = True
            base = SVC(**_params)
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            cat_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer(
                transformers=[
                    ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                ],
                remainder='drop'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        if model_type == 'mlp':
            # Convert hidden_layer_sizes from list to tuple for sklearn
            mlp_params = params.copy() if params else {}
            if 'hidden_layer_sizes' in mlp_params and isinstance(mlp_params['hidden_layer_sizes'], list):
                mlp_params['hidden_layer_sizes'] = tuple(mlp_params['hidden_layer_sizes'])
            if regression:
                # Wrap MLPRegressor in a preprocessing pipeline for robust handling of mixed types
                num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
                cat_pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                pre = ColumnTransformer(
                    transformers=[
                        ('num', num_pipe, make_column_selector(dtype_include=['number'])),
                        ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))
                    ],
                    remainder='drop'
                )
                base = MLPRegressor(**mlp_params)
                return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
            base = MLPClassifier(**mlp_params)
            pre = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_include=['object', 'category', 'string', 'bool']))],
                remainder='passthrough'
            )
            return Pipeline(steps=[('preprocess', pre), ('estimator', base)])
        raise ValueError(f"Unknown model_type: {model_type}")