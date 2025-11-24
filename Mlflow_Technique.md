# 📘 MLflow Complete Revision Guide

## 1. Core Components Overview

**MLflow** = Open-source platform for complete ML lifecycle management

**4 Main Components:**
- **Tracking** – Log experiments (params, metrics, artifacts)
- **Projects** – Package reproducible code
- **Models** – Deploy models uniformly
- **Registry** – Version & manage production models

---

## 2. Installation & Setup

```bash
# Install
pip install mlflow
pip install mlflow[extras]  # with all dependencies

# Start UI
mlflow ui
mlflow ui --port 5000 --host 0.0.0.0

# Start server with backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
```

```python
# Configure tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("experiment_name")
```

---

## 3. MLflow Tracking

### 3.1 Basic Workflow
```python
import mlflow

with mlflow.start_run(run_name="my_run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
```

### 3.2 Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `start_run()` | Start new run | `mlflow.start_run(run_name="test")` |
| `end_run()` | End current run | `mlflow.end_run()` |
| `log_param()` | Log single parameter | `mlflow.log_param("epochs", 100)` |
| `log_params()` | Log multiple parameters | `mlflow.log_params({"lr": 0.01, "batch": 32})` |
| `log_metric()` | Log single metric | `mlflow.log_metric("loss", 0.5, step=1)` |
| `log_metrics()` | Log multiple metrics | `mlflow.log_metrics({"acc": 0.9, "f1": 0.85})` |
| `log_artifact()` | Log file | `mlflow.log_artifact("plot.png")` |
| `log_artifacts()` | Log directory | `mlflow.log_artifacts("outputs/")` |
| `log_dict()` | Log dict as JSON/YAML | `mlflow.log_dict(config, "config.json")` |
| `log_text()` | Log text content | `mlflow.log_text("info", "log.txt")` |
| `log_figure()` | Log matplotlib figure | `mlflow.log_figure(fig, "plot.png")` |
| `log_image()` | Log image | `mlflow.log_image(image, "img.png")` |
| `set_tag()` | Add metadata tag | `mlflow.set_tag("model_type", "RF")` |
| `set_tags()` | Add multiple tags | `mlflow.set_tags({"team": "ds", "version": "v1"})` |

### 3.3 Run Management
```python
# Start run with options
run = mlflow.start_run(run_name="my_run", experiment_id="1", nested=True)

# Get active run
active_run = mlflow.active_run()

# Get run details
run_data = mlflow.get_run(run_id)

# Nested runs
with mlflow.start_run():
    with mlflow.start_run(nested=True):
        # Child run
        pass
```

### 3.4 Experiment Management
```python
# Create experiment
exp_id = mlflow.create_experiment(
    "my_exp",
    artifact_location="s3://bucket",
    tags={"version": "1.0"}
)

# Get experiment
exp = mlflow.get_experiment(experiment_id)
exp = mlflow.get_experiment_by_name("my_exp")

# List experiments
experiments = mlflow.search_experiments()

# Set experiment
mlflow.set_experiment("experiment_name")

# Delete/Restore
mlflow.delete_experiment(experiment_id)
mlflow.restore_experiment(experiment_id)
```

### 3.5 Searching Runs
```python
# Search runs
runs = mlflow.search_runs(
    experiment_ids=["1", "2"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Filter examples
"metrics.accuracy > 0.9"
"params.model = 'RandomForest' and metrics.accuracy > 0.85"
"tags.team = 'ml_team'"
"attributes.status = 'FINISHED'"
```

### 3.6 Autologging
```python
# Enable for all frameworks
mlflow.autolog()

# Framework-specific
mlflow.sklearn.autolog()
mlflow.pytorch.autolog()
mlflow.tensorflow.autolog()
mlflow.keras.autolog()
mlflow.xgboost.autolog()
mlflow.lightgbm.autolog()

# Disable
mlflow.autolog(disable=True)

# Configure
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)
```

---

## 4. MLflow Projects

### 4.1 MLproject File Structure
```yaml
name: My_Project

conda_env: conda.yaml
# or
docker_env:
  image: my-docker-image

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
      l1_ratio: {type: float, default: 0.1}
    command: "python train.py --alpha {alpha} --l1-ratio {l1_ratio}"
  
  validate:
    parameters:
      model_path: path
    command: "python validate.py --model-path {model_path}"
```

### 4.2 Running Projects
```python
# Run local project
mlflow.run(
    ".",
    parameters={"alpha": 0.5, "l1_ratio": 0.1},
    experiment_name="my_experiment"
)

# Run from Git
mlflow.run(
    "https://github.com/user/repo",
    version="main",
    parameters={"param": "value"}
)

# Run specific entry point
mlflow.run(".", entry_point="validate", parameters={"model_path": "path"})
```

### 4.3 CLI Commands
```bash
# Run project
mlflow run . -P alpha=0.5 -P l1_ratio=0.1

# Run from Git
mlflow run https://github.com/user/repo -v main

# Run specific entry point
mlflow run . -e validate -P model_path=/path/to/model
```

---

## 5. MLflow Models

### 5.1 Model Flavors

| Flavor | Log Function | Load Function |
|--------|-------------|---------------|
| Scikit-learn | `mlflow.sklearn.log_model()` | `mlflow.sklearn.load_model()` |
| PyTorch | `mlflow.pytorch.log_model()` | `mlflow.pytorch.load_model()` |
| TensorFlow | `mlflow.tensorflow.log_model()` | `mlflow.tensorflow.load_model()` |
| Keras | `mlflow.keras.log_model()` | `mlflow.keras.load_model()` |
| XGBoost | `mlflow.xgboost.log_model()` | `mlflow.xgboost.load_model()` |
| LightGBM | `mlflow.lightgbm.log_model()` | `mlflow.lightgbm.load_model()` |
| Generic | `mlflow.pyfunc.log_model()` | `mlflow.pyfunc.load_model()` |

### 5.2 Logging Models

#### Scikit-learn
```python
import mlflow.sklearn

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=signature,
    input_example=X_train[:5]
)

# Save model
mlflow.sklearn.save_model(model, "path/to/model")

# Load model
loaded_model = mlflow.sklearn.load_model("runs:/run_id/model")
```

#### PyTorch
```python
import mlflow.pytorch

mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    signature=signature
)
```

#### TensorFlow/Keras
```python
import mlflow.tensorflow
import mlflow.keras

mlflow.keras.log_model(
    model=keras_model,
    artifact_path="model",
    signature=signature
)
```

#### Custom Python Models
```python
import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        self.model = pickle.load(open(context.artifacts["model_path"], "rb"))
    
    def predict(self, context, model_input):
        return self.model.predict(model_input)

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=CustomModel(),
    artifacts={"model_path": "local/path/to/model.pkl"}
)
```

### 5.3 Model Signatures
```python
from mlflow.models import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Infer signature automatically
signature = infer_signature(X_train, model.predict(X_train))

# Define signature manually
input_schema = Schema([
    ColSpec("double", "feature1"),
    ColSpec("double", "feature2")
])
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log with signature
mlflow.sklearn.log_model(model, "model", signature=signature)
```

### 5.4 Loading Models
```python
# Load as Python function
model = mlflow.pyfunc.load_model("runs:/run_id/model")
model = mlflow.pyfunc.load_model("models:/model_name/version")
model = mlflow.pyfunc.load_model("models:/model_name/Production")

# Load native format
sklearn_model = mlflow.sklearn.load_model("runs:/run_id/model")

# Make predictions
predictions = model.predict(data)
```

### 5.5 Model Deployment

#### Serve Model Locally
```bash
# Serve model via REST API
mlflow models serve -m runs:/run_id/model -p 5001
mlflow models serve -m models:/model_name/Production

# Test the endpoint
curl -X POST -H "Content-Type: application/json" \
  --data '{"dataframe_split": {"columns":["x"], "data":[[1]]}}' \
  http://localhost:5001/invocations
```

#### Build Docker Image
```bash
mlflow models build-docker -m runs:/run_id/model -n my-model-image
```

#### Batch Predictions
```bash
mlflow models predict -m runs:/run_id/model -i input.csv -o output.csv
```

#### Deploy to Cloud
```python
# Deploy to SageMaker
import mlflow.sagemaker
mlflow.sagemaker.deploy(
    app_name="my-app",
    model_uri="runs:/run_id/model",
    region_name="us-west-2"
)

# Deploy to Azure ML
import mlflow.azureml
mlflow.azureml.deploy(
    model_uri="runs:/run_id/model",
    workspace=workspace,
    deployment_name="my-deployment"
)
```

---

## 6. MLflow Model Registry

### 6.1 Registering Models
```python
# Register during logging
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="my_model"
)

# Register existing run
result = mlflow.register_model(
    model_uri="runs:/run_id/model",
    name="my_model"
)

# Using client
from mlflow import MlflowClient
client = MlflowClient()
client.create_registered_model(
    name="my_model",
    tags={"task": "classification"},
    description="My classification model"
)
```

### 6.2 Managing Model Versions
```python
client = MlflowClient()

# Create model version
version = client.create_model_version(
    name="my_model",
    source="runs:/run_id/model",
    run_id="run_id"
)

# Get model version
version = client.get_model_version(name="my_model", version="1")

# Update model version
client.update_model_version(
    name="my_model",
    version="1",
    description="Updated description"
)

# Search model versions
versions = client.search_model_versions("name='my_model'")

# Delete model version
client.delete_model_version(name="my_model", version="1")
```

### 6.3 Model Stages
```python
# Transition model to stage
client.transition_model_version_stage(
    name="my_model",
    version="1",
    stage="Production"
)

# Available stages: None, Staging, Production, Archived

# Load model by stage
model = mlflow.pyfunc.load_model("models:/my_model/Production")
model = mlflow.pyfunc.load_model("models:/my_model/Staging")
```

### 6.4 Model Aliases (Modern Approach)
```python
# Set alias
client.set_registered_model_alias(
    name="my_model",
    alias="champion",
    version="3"
)

# Get model by alias
client.get_model_version_by_alias(name="my_model", alias="champion")

# Load model by alias
model = mlflow.pyfunc.load_model("models:/my_model@champion")

# Delete alias
client.delete_registered_model_alias(name="my_model", alias="champion")
```

### 6.5 Model Metadata
```python
# Add tags to model version
client.set_model_version_tag(
    name="my_model",
    version="1",
    key="validation_status",
    value="approved"
)

# Add tags to registered model
client.set_registered_model_tag(
    name="my_model",
    key="task",
    value="classification"
)

# Get registered model
model = client.get_registered_model("my_model")

# Update registered model
client.update_registered_model(
    name="my_model",
    description="Updated description"
)

# Rename registered model
client.rename_registered_model(
    name="my_model",
    new_name="my_model_v2"
)
```

---

## 7. Advanced Features

### 7.1 MLflow Client
```python
from mlflow import MlflowClient

client = MlflowClient()

# Run operations
run = client.get_run(run_id)
client.log_param(run_id, "param", value)
client.log_metric(run_id, "metric", value)
client.set_tag(run_id, "tag", value)
client.log_artifact(run_id, "local_path")

# Experiment operations
experiment = client.get_experiment(experiment_id)
experiments = client.search_experiments()
client.create_experiment("name")

# Search runs
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.rmse < 0.9",
    order_by=["metrics.rmse ASC"]
)
```

### 7.2 System Metrics Logging
```python
# Enable system metrics logging
mlflow.enable_system_metrics_logging()

# Logged metrics include:
# - cpu_utilization_percentage
# - system_memory_usage_megabytes
# - disk_usage_percentage
# - network_receive_megabytes
# - network_transmit_megabytes
```

### 7.3 Dataset Logging
```python
from mlflow.data.pandas_dataset import PandasDataset

# Create dataset
dataset = mlflow.data.from_pandas(
    df,
    source="path/to/data.csv",
    name="training_data"
)

# Log dataset
mlflow.log_input(dataset, context="training")
mlflow.log_input(dataset, context="validation")
```

### 7.4 Model Evaluation
```python
# Evaluate model
result = mlflow.evaluate(
    model="runs:/run_id/model",
    data=test_data,
    targets="target_column",
    model_type="classifier",
    evaluators="default"
)

# Custom evaluator
from mlflow.metrics import make_metric

def custom_metric(eval_df, builtin_metrics):
    return eval_df["prediction"].mean()

custom = make_metric(
    eval_fn=custom_metric,
    greater_is_better=True,
    name="custom_metric"
)

result = mlflow.evaluate(
    model=model,
    data=data,
    targets="target",
    extra_metrics=[custom]
)
```

---

## 8. Essential CLI Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `mlflow ui` | Launch tracking UI | `mlflow ui --port 5000` |
| `mlflow server` | Start tracking server | `mlflow server --host 0.0.0.0` |
| `mlflow run` | Execute project | `mlflow run . -P alpha=0.5` |
| `mlflow experiments create` | Create experiment | `mlflow experiments create -n "MyExp"` |
| `mlflow experiments list` | List experiments | `mlflow experiments list` |
| `mlflow experiments delete` | Delete experiment | `mlflow experiments delete <id>` |
| `mlflow runs list` | List runs | `mlflow runs list --experiment-id 1` |
| `mlflow models serve` | Serve model | `mlflow models serve -m <uri> -p 5001` |
| `mlflow models predict` | Batch predictions | `mlflow models predict -m <uri> -i input.csv` |
| `mlflow models build-docker` | Build Docker image | `mlflow models build-docker -m <uri>` |
| `mlflow models list` | List registered models | `mlflow models list` |

---

## 9. Best Practices

### ✅ Organization
- Use meaningful experiment names for logical grouping
- Apply consistent naming conventions for parameters and metrics
- Tag runs with metadata (`team`, `version`, `git_commit`) for easy filtering
- Document experiments with clear descriptions

### ✅ Tracking
- Log all important parameters, metrics, and artifacts
- Use nested runs for complex pipelines and hyperparameter tuning
- Version your data (log data hashes or versions)
- Track data lineage by logging dataset information
- Use `log_metrics()` for batch logging to improve performance

### ✅ Model Management
- Always include model signatures to define input/output schemas
- Provide input examples to help with model serving
- Use Model Registry for production model management
- Test models thoroughly before promoting to Production stage
- Apply semantic versioning concepts

### ✅ Performance
- Batch metric logging with `log_metrics()` instead of multiple `log_metric()` calls
- Limit artifact size - don't log unnecessarily large files
- Use remote tracking server for team collaboration
- Archive old experiments to keep workspace clean

### ✅ Production Code Example
```python
def train_model(params):
    mlflow.set_experiment("production_training")
    
    with mlflow.start_run(run_name=f"{params['model']}_v1"):
        # Log parameters
        mlflow.log_params(params)
        
        # Log code version
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tags({
            "team": "data_science",
            "environment": "production"
        })
        
        # Train model
        model = train(params)
        
        # Evaluate
        metrics = evaluate(model)
        mlflow.log_metrics(metrics)
        
        # Log model with signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name="prod_model"
        )
        
        return model
```

---

## 10. Quick Reference Card

### Core Concepts
- **Experiment** – Collection of related runs
- **Run** – Single execution with logged data
- **Artifact** – Output files (models, plots, datasets)
- **Parameter** – Input configuration (hyperparameters)
- **Metric** – Evaluation results (accuracy, loss)
- **Tag** – Metadata for organization and search

### Most Used Functions
```python
# Setup
mlflow.set_tracking_uri(uri)
mlflow.set_experiment(name)

# Tracking
with mlflow.start_run(run_name="name"):
    mlflow.log_params({"lr": 0.01, "epochs": 100})
    mlflow.log_metrics({"acc": 0.95, "loss": 0.05})
    mlflow.log_artifact("plot.png")
    mlflow.sklearn.log_model(model, "model")

# Registry
mlflow.register_model("runs:/run_id/model", "model_name")
client.transition_model_version_stage(name, version, stage)

# Loading
model = mlflow.pyfunc.load_model("models:/model_name/Production")
predictions = model.predict(data)

# Search
runs = mlflow.search_runs(filter_string="metrics.accuracy > 0.9")
```

### Filter String Syntax
```python
# Metrics
"metrics.accuracy > 0.9"
"metrics.loss < 0.1"

# Parameters
"params.model = 'RandomForest'"
"params.learning_rate < 0.01"

# Combined
"params.model = 'XGBoost' and metrics.f1 > 0.85"

# Tags
"tags.team = 'ml_team'"
"tags.environment = 'production'"

# Attributes
"attributes.status = 'FINISHED'"
"attributes.run_name LIKE 'experiment_%'"
```

### Model URI Formats
```python
# By run ID
"runs:/run_id/model"

# By model version
"models:/model_name/1"

# By stage
"models:/model_name/Production"
"models:/model_name/Staging"

# By alias
"models:/model_name@champion"
```

---

## 11. Learning Path

### Week 1-2: Basics
- Install MLflow and explore UI
- Log parameters, metrics, and artifacts
- Try autologging with sklearn, pytorch, tensorflow
- Practice with simple classification/regression tasks

### Week 3-4: Intermediate
- Create MLflow Projects with MLproject files
- Work with Model Registry (register, version, stage)
- Implement model deployment (serve locally)
- Build end-to-end tracking workflow

### Week 5-6: Advanced
- Create custom model flavors with pyfunc
- Learn production deployment patterns
- Integrate with CI/CD pipelines
- Implement model evaluation and monitoring

### Week 7-8: Real Projects
- Build complete ML pipeline with MLflow
- Implement model versioning strategy
- Set up production model serving
- Create model monitoring dashboard

### Top Learning Resources

**Official Documentation:**
- MLflow Docs: https://mlflow.org/docs/latest/
- API Reference: https://mlflow.org/docs/latest/python_api/
- GitHub: https://github.com/mlflow/mlflow

**Video Tutorials (Best to Worst):**
1. **Databricks YouTube Channel** – Official MLflow tutorials and production examples
2. **Krish Naik MLflow Series** – Beginner-friendly end-to-end projects
3. **Official MLflow Channel** – Webinars and feature deep-dives
4. **AssemblyAI** – Quick crash courses
5. **Weights & Biases** – Comparative ML experiment tracking tutorials

**Online Courses:**
1. **Databricks Academy** – Free MLflow courses (production-focused)
2. **Coursera MLOps Specialization** – Includes MLflow components
3. **DataCamp** – "Introduction to MLflow" (interactive)
4. **Udemy** – "MLflow in Action" and "MLOps: Development to Production"

**Books:**
1. "Practical MLOps" by Noah Gift & Alfredo Deza
2. "Introducing MLOps" by Mark Treveil et al.
3. "Machine Learning Engineering" by Andriy Burkov

**Practice Projects:**
- **Beginner:** Simple model tracking, parameter tuning comparison
- **Intermediate:** End-to-end pipeline, model registry workflow
- **Advanced:** Multi-stage deployment, A/B testing, custom metrics

---

## 12. Common Use Cases

### Use Case 1: Hyperparameter Tuning
```python
for lr in [0.001, 0.01, 0.1]:
    with mlflow.start_run():
        mlflow.log_param("learning_rate", lr)
        model = train_model(lr)
        accuracy = evaluate(model)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
```

### Use Case 2: Model Comparison
```python
models = ["RandomForest", "XGBoost", "LightGBM"]

for model_name in models:
    with mlflow.start_run(run_name=model_name):
        model = train(model_name)
        metrics = evaluate(model)
        mlflow.log_params({"model_type": model_name})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
```

### Use Case 3: Production Pipeline
```python
def production_pipeline():
    # Train
    with mlflow.start_run(run_name="training"):
        model = train_model()
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name="prod_model"
        )
        run_id = mlflow.active_run().info.run_id
    
    # Register
    client = MlflowClient()
    version = client.get_latest_versions("prod_model")[0].version
    
    # Validate
    if validate_model(model):
        client.transition_model_version_stage(
            name="prod_model",
            version=version,
            stage="Production"
        )
```

---

## Summary

MLflow provides a complete platform for ML lifecycle management with four main components:

1. **Tracking** – Log and query experiments (foundation)
2. **Projects** – Package reproducible code
3. **Models** – Deploy models uniformly
4. **Registry** – Version and manage production models

**Key Takeaways:**
- Start with Tracking – it's the foundation
- Always use Model Registry for production models
- Implement proper logging from day one
- Follow best practices for team collaboration
- Practice with real-world projects
- Master components in order: Tracking → Projects → Models → Registry

**Success Formula:**
```
Good Tracking + Model Registry + Best Practices = Production-Ready MLflow
```

Good luck with your MLflow journey! 🚀