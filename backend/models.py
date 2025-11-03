from extensions import db
from flask_login import UserMixin

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    file_path = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    description = db.Column(db.Text, nullable=True)
    data_source = db.Column(db.Text, nullable=False)  # required: where the data came from
    license_info = db.Column(db.Text, nullable=False)  # required: license terms
    # NOTE: keep model fields in sync with the existing DB schema.
    # Historical schema uses `target_feature` and `train_test_split`.
    # Avoid adding `target_variable`/`split_percent` here unless you run a DB migration.
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=True)
    input_features = db.Column(db.String(500), nullable=True)  # comma-separated feature names
    target_feature = db.Column(db.String(100), nullable=True)
    regression = db.Column(db.Boolean, default=False, nullable=True) # True if regression, False if classification
    train_test_split = db.Column(db.Float, nullable=True)
    
class ModelEntry(db.Model):
    """
    Stores models in DB. model_blob contains a pickled model object.
    params and metrics are pickled dicts.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=True)
    model_type = db.Column(db.String(80), nullable=False)
    # params and metrics stored as JSON strings
    params = db.Column(db.String(500), nullable=True)
    metrics = db.Column(db.String(500), nullable=True)
    model_blob = db.Column(db.LargeBinary, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=True)
    early_stopped = db.Column(db.Boolean, default=False, nullable=True)
    current_epoch = db.Column(db.Integer, nullable=True)  # Track last completed epoch for resuming