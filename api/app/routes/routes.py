
from fastapi import APIRouter
from app.logging.logging_config import setup_logger
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from huggingface_hub import HfApi, hf_hub_download
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import joblib
import io
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, r2_score, silhouette_score, classification_report, mean_squared_error
from app.components.Upload.data_upload import upload_data
from app.components.Upload.upload import upload
from app.components.Cleaning.clean import clean_data
from app.utils.session import SESSIONS
from fastapi import Body    
import json
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
import os
logger = setup_logger(__name__)

router = APIRouter()
load_dotenv()

HF_TOKEN = os.getenv("hf_token")
HF_REPO_ID = "prthm20/ZeoMl"

api = HfApi()

@router.get("/")
async def home():
    logger.info("Home route accessed")
    return {"message": "ZeroMl is live!"}


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        res = await upload_data(file)

        return res

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get-file")
async def get_file(filename: str):
    from app.components.ExtractFile.extract import get_file as extract_file
    try:
        res = await extract_file(filename)

        return res

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_2(file: UploadFile = File(...)):
    try:
        res = await upload(file)

        return res

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clean-data")
async def clean(session_id: str = Body(...), instruction: str = Body(...)):
    try:
        res = await clean_data(session_id, instruction)

        return res

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save-cleaned-file")
async def save_cleaned_file(session_id: str = Body(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    
    df = SESSIONS[session_id]
    filename = f"cleaned_{session_id}.csv"
    df.to_csv(filename, index=False)

    return {"message": "File saved successfully", "path": filename}





@router.post("/train-model")
async def train_model(
    session_id: str = Form(...),
    target: str = Form(None),
    model_choice: str = Form(None),
    params: str = Form(None)
):
    """
    Robust train_model: strong preprocessing & target encoding + debug logs.
    """
    
    try:
        import json
        from sklearn.preprocessing import LabelEncoder

        # === Read file ===
        repo_id="prthm20/ZeoMl"
        filename=f"{session_id}_cleaned.csv"
        print(HF_TOKEN)
        file_path= hf_hub_download(repo_id=repo_id, filename=filename,repo_type="dataset", token=HF_TOKEN)  
        df = pd.read_csv(file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

        # === Determine target column (auto if not provided) ===
        target_col = target if target and target in df.columns else df.columns[-1]

        # Quick debug: show columns and first rows
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Using target column: {target_col}")
        logger.info(f"Head of dataframe:\n{df.head(5)}")

        # === Split X and y ===
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # === Robust feature preprocessing ===
        # 1) Convert object columns to category to ensure get_dummies works properly
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = X[col].astype("category")

        # 2) One-hot encode categorical columns
        X = pd.get_dummies(X, drop_first=True)

        # 3) Force all columns to numeric where possible; non-convertible -> NaN
        X = X.apply(lambda col: pd.to_numeric(col, errors="coerce"))

        # 4) Fill NaNs with median of column (numeric_only)
        X = X.fillna(X.median(numeric_only=True))

        # Debug: show types and sample after preprocessing
        logger.info("Feature dtypes after preprocessing:")
        logger.info(X.dtypes.head(20).to_dict())
        logger.info("Feature sample after preprocessing:")
        logger.info(X.head(5).to_dict())

        # === Determine problem type robustly ===
        # If target is object -> classification
        if y.dtype == "object":
            problem_type = "classification"
        elif np.issubdtype(y.dtype, np.integer) and len(y.unique()) <= 5:
            # small integer set -> classification
            problem_type = "classification"
        else:
            problem_type = "regression"

        # === Encode target for classification if needed ===
        if problem_type == "classification":
            # Debug: show unique values BEFORE encoding
            logger.info(f"Target dtype before encoding: {y.dtype}")
            logger.info(f"Target unique values (sample up to 20): {list(y.unique()[:20])}")

            # If target is object or not numeric, label encode
            if y.dtype == "object" or not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))  # ensure strings
                y = pd.Series(y_encoded, index=y.index)
                logger.info(f"Label encoding applied. Classes: {list(le.classes_)}")
            else:
                # numeric dtype but maybe floats that represent classes
                # convert floats that are whole numbers to ints
                if np.issubdtype(y.dtype, np.floating) and np.all(np.mod(y.dropna(), 1) == 0):
                    y = y.astype(int)
                # otherwise leave numeric as-is (may be problematic — but we'll proceed)

            logger.info(f"Target after encoding dtype: {y.dtype}, unique: {list(pd.Series(y).unique()[:20])}")

        # === Final train/test split ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Debug: shapesx
        logger.info(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}")

        # === Parse hyperparameters safely ===
        hyperparams = {}
        if params:
            try:
                hyperparams = json.loads(params)
                if not isinstance(hyperparams, dict):
                    hyperparams = {}
            except Exception:
                # if params is not valid JSON, ignore but log
                logger.warning(f"Could not parse params JSON: {params}")
                hyperparams = {}

        # === Default and final params (merge) ===
        default_params = {
            "RandomForestClassifier": {"random_state": 42},
            "LogisticRegression": {"max_iter": 1000, "random_state": 42},
            "RandomForestRegressor": {"random_state": 42},
            "LinearRegression": {},
            "KMeans": {"random_state": 42}
        }
        final_params = {**default_params.get(model_choice or "", {}), **hyperparams}

        # === Initialize model safely ===
        if model_choice == "RandomForestClassifier":
            model = RandomForestClassifier(**final_params)
        elif model_choice == "LogisticRegression":
            model = LogisticRegression(**final_params)
        elif model_choice == "RandomForestRegressor":
            model = RandomForestRegressor(**final_params)
        elif model_choice == "LinearRegression":
            model = LinearRegression(**final_params)
        elif model_choice == "KMeans":
            model = KMeans(**final_params)
        else:
            # fallback: choose based on problem type
            if problem_type == "classification":
                model = RandomForestClassifier(**final_params)
                model_choice = "RandomForestClassifier"
            else:
                model = RandomForestRegressor(**final_params)
                model_choice = "RandomForestRegressor"

        # === Fit model with try/except to capture the exact failing array ===
        try:
            model.fit(X_train, y_train)
        except Exception as fit_exc:
            # dump some debug info and re-raise a readable HTTPException
            # Check types in X_train and y_train samples
            dbg = {
                "X_train_dtypes": X_train.dtypes.apply(lambda t: str(t)).to_dict(),
                "X_train_sample_first_row": X_train.iloc[0].to_dict() if X_train.shape[0] > 0 else {},
                "y_train_sample_first": str(y_train.iloc[0]) if y_train.shape[0] > 0 else None,
                "error": str(fit_exc)
            }
            logger.error(f"Model fit failed: {dbg}")
            raise HTTPException(status_code=500, detail=f"Model fit failed: {fit_exc} -- debug: {dbg}")

        # === Predict & metrics ===
        metrics = {}
        if model_choice == "KMeans":
            metrics["inertia"] = float(model.inertia_)
            metrics["n_clusters"] = getattr(model, "n_clusters", None)
        else:
            preds = model.predict(X_test)

            if problem_type == "classification":
                # If classifier outputs probabilities / multi-dim array, convert to labels
                if hasattr(preds, "shape") and getattr(preds, "ndim", 1) > 1 and preds.shape[1] > 1:
                    preds_labels = np.argmax(preds, axis=1)
                else:
                    # If preds are floats (e.g., sklearn might output floats for some models),
                    # round to nearest integer class
                    preds_labels = np.round(preds).astype(int) if np.issubdtype(preds.dtype, np.floating) else preds

                metrics["accuracy"] = float(accuracy_score(y_test, preds_labels))
                metrics["classification_report"] = classification_report(y_test, preds_labels, output_dict=True)
            else:
                metrics["r2_score"] = float(r2_score(y_test, preds))
                metrics["mse"] = float(mean_squared_error(y_test, preds))

        # === Save model ===
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/{(model_choice or 'model')}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        

        try:
           repo_id = "prthm20/ZeoMl"  # your dataset repo
           hf_filename = f"{session_id}_{model_choice}_{timestamp}.pkl"

    # Upload the model file directly from disk
           api.upload_file(
             path_or_fileobj=model_path,          # ✅ path to your .pkl file
             path_in_repo=hf_filename,            # file name in the repo
             repo_id=repo_id,
             repo_type="dataset",                 # ✅ important (same repo type)
             commit_message=f"Trained model for session {session_id}"
         )
           hf_download_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{hf_filename}"

           hf_status = f"Model uploaded to Hugging Face: {repo_id}/{hf_filename}"
        except Exception as e:
                hf_status = f"Error uploading model to Hugging Face: {e}"
        return {
            "status": "success",
            "problem_type": problem_type,
            "target_column": target_col,
            "model_name": model_choice,
            "hyperparameters_used": final_params,
            "metrics": metrics,
            "hf_filename" : hf_filename,
            "huggingface_download_url": hf_download_url,
            "model_path": model_path.replace("\\", "/")
        }

    except HTTPException:
        # re-raise HTTPExceptions unchanged
        raise
    except Exception as e:
        logger.exception("Unhandled training error")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

# ✅ Ensure this line is SEPARATED by a blank line

@router.get("/hyperparameters")
async def get_hyperparameters(model_name: str):
    """
    Return default hyperparameters for a given model.
    """
    default_hyperparameters = {
        "RandomForestClassifier": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        },
        "LogisticRegression": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42
        },
        "RandomForestRegressor": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        },
        "LinearRegression": {
            "fit_intercept": True,
            "copy_X": True,
        },
        "KMeans": {
            "n_clusters": 8,
            "init": "k-means++",
            "max_iter": 300,
            "random_state": 42
        }
    }

    if model_name not in default_hyperparameters:
        raise HTTPException(status_code=400, detail="Unsupported model name.")

    return {"default_hyperparameters": default_hyperparameters[model_name]}


@router.get("/download-model")
async def download_model(filename: str):
    """
    Download a trained model (.pkl) from Hugging Face and return it as a downloadable file.
    Example:
    /download-model?filename=139e5062-558a-4052-838b-4ad316c5878a_RandomForestClassifier_20251101_160025.pkl
    """
    try:
        repo_id = "prthm20/ZeoMl"  # your dataset repo
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN
        )

        if not os.path.exists(local_path):
            raise HTTPException(status_code=404, detail="Model file not found.")

        return FileResponse(
            path=local_path,
            filename=filename,
            media_type="application/octet-stream"
        )

    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")
