"""
Script Name: main.py
Author: Shane Lee
Description: Cloud Training Worker. Implements Stateless Feature Extraction (Feature Hashing) 
             and Incremental Learning. Generates Executive Business Intelligence 
             artifacts (Priority Alerts, Sentiment Drivers, Forensic Audit).
"""

import os
import sys
import logging
import argparse
import joblib
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import gc
import sklearn
from time import time

from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
)
from sklearn.model_selection import train_test_split

try:
    import scalability_utils as utils
except ImportError:
    print("CRITICAL: scalability_utils.py not found.")
    sys.exit(1)

# --- Forensic Logging ---
os.makedirs("outputs", exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler("outputs/forensic_audit.log")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger = logging.getLogger("ForensicLogger")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def generate_keyword_insights(text_list, labels, top_n=20):
    """
    Extracts top keywords for Positive vs Negative reviews from the test sample.
    Uses a standard CountVectorizer since the test sample is small enough for RAM.
    """
    try:
        df = pd.DataFrame({'text': text_list, 'label': labels})
        
        # Split by sentiment
        pos_text = df[df['label'] == 1]['text']
        neg_text = df[df['label'] == 0]['text']
        
        insights = []
        
        # Helper to get top words
        def get_top_words(corpus, sentiment_name):
            if len(corpus) == 0: return
            # Use bi-grams (2 words) to capture context like "not good"
            cv = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=top_n)
            X = cv.fit_transform(corpus)
            counts = X.sum(axis=0).A1
            vocab = cv.get_feature_names_out()
            
            # Zip and sort
            freqs = sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True)
            
            for word, count in freqs:
                insights.append({
                    'Sentiment_Segment': sentiment_name,
                    'Keyword_Phrase': word,
                    'Frequency': count
                })

        get_top_words(pos_text, "Positive_Drivers")
        get_top_words(neg_text, "Negative_Drivers")
        
        return pd.DataFrame(insights)
    except Exception as e:
        logger.warning(f"Could not generate keyword insights: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--n_features", type=int, default=1048576)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--text_col", type=str, default="Text")
    parser.add_argument("--label_col", type=str, default="Score")
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    mlflow.start_run()
    logger.info(f"Loading stream: {args.data} (Batch: {args.batch_size})")
    
    # --- Pipeline Construction ---
    vectorizer = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        norm='l2',
        binary=False
    )

    clf = PassiveAggressiveClassifier(
        C=1.0,
        loss='hinge',
        random_state=42,
        warm_start=True,
        n_jobs=1
    )
    
    classes = np.array([0, 1])
    
    # Accumulators for the Final Epoch Forensic Report
    report_text = []
    report_y_true = []
    report_y_pred = []
    report_confidence = []
    
    # --- Streaming Training Loop ---
    try:
        for epoch in range(args.epochs):
            logger.info(f"--- Starting Epoch {epoch + 1}/{args.epochs} ---")
            
            reader = pd.read_csv(args.data, chunksize=args.batch_size, on_bad_lines='skip')
            
            for chunk in reader:
                # A. Validation
                if args.text_col not in chunk.columns or args.label_col not in chunk.columns:
                    continue
                
                chunk = chunk.dropna(subset=[args.text_col, args.label_col])
                if chunk.empty: continue

                # B. Target Transformation
                y_chunk = (chunk[args.label_col] > args.threshold).astype(int)
                X_chunk = chunk[args.text_col].astype(str)

                # C. Train/Test Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_chunk, y_chunk, test_size=0.2, random_state=42
                )
                
                # D. Incremental Learning
                X_train_vec = vectorizer.transform(X_train)
                clf.partial_fit(X_train_vec, y_train, classes=classes)
                
                # E. Accumulate Evaluation Data (Final Epoch Only)
                if epoch == args.epochs - 1:
                    X_test_vec = vectorizer.transform(X_test)
                    preds = clf.predict(X_test_vec)
                    conf = clf.decision_function(X_test_vec) # Signed distance
                    
                    report_text.extend(X_test.tolist())
                    report_y_true.extend(y_test.tolist())
                    report_y_pred.extend(preds.tolist())
                    report_confidence.extend(conf.tolist())
                
                del chunk, X_train, X_test, y_train, y_test, X_train_vec
                gc.collect()

    except Exception as e:
        logger.error(f"CRITICAL: Stream processing failed. Error: {e}")
        sys.exit(1)

    # --- Metrics Calculation ---
    logger.info("Calculating Comprehensive Metrics...")
    
    if len(report_y_true) > 0:
        acc = accuracy_score(report_y_true, report_y_pred)
        f1 = f1_score(report_y_true, report_y_pred, average='weighted')
        prec = precision_score(report_y_true, report_y_pred, average='weighted', zero_division=0)
        rec = recall_score(report_y_true, report_y_pred, average='weighted', zero_division=0)
        try:
            auc = roc_auc_score(report_y_true, report_confidence)
        except ValueError:
            auc = 0.0

        logger.info(f"FINAL METRICS: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # --- GENERATE EXECUTIVE ARTIFACTS ---
        logger.info("Generating Executive Artifacts...")
        
        # 1. Master DataFrame
        full_df = pd.DataFrame({
            'Review_Text': report_text,
            'Actual_Sentiment': report_y_true,
            'Predicted_Sentiment': report_y_pred,
            'Confidence_Score': np.abs(report_confidence), # Absolute value for magnitude
            'Raw_Decision': report_confidence # Keep signed for debugging
        })
        
        # 2. Artifact A: Priority Alerts (High Confidence Negatives)
        # Logic: Predicted Negative (0) AND High Confidence
        priority_df = full_df[
            (full_df['Predicted_Sentiment'] == 0)
        ].sort_values(by='Confidence_Score', ascending=False).head(200)
        
        priority_path = "outputs/priority_alerts.csv"
        priority_df.to_csv(priority_path, index=False)
        logger.info(f"Generated Priority Alerts: {priority_path}")
        mlflow.log_artifact(priority_path)

        # 3. Artifact B: Sentiment Drivers (Keyword Analysis)
        drivers_df = generate_keyword_insights(report_text, report_y_true)
        drivers_path = "outputs/sentiment_drivers.csv"
        drivers_df.to_csv(drivers_path, index=False)
        logger.info(f"Generated Sentiment Drivers: {drivers_path}")
        mlflow.log_artifact(drivers_path)

        # 4. Artifact C: Full Forensic Audit
        audit_path = "outputs/forensic_audit_full.csv"
        full_df.to_csv(audit_path, index=False)
        logger.info(f"Generated Full Audit: {audit_path}")
        mlflow.log_artifact(audit_path)

    else:
        logger.warning("No test data accumulated. Metrics skipped.")

    # --- Model Serialisation ---
    logger.info("Serialising model pipeline...")
    final_pipeline = make_pipeline(vectorizer, clf)
    
    artifact = {
        'pipeline': final_pipeline,
        'metadata': {
            'python': sys.version,
            'sklearn': sklearn.__version__,
            'metrics': {'accuracy': acc, 'auc': auc},
            'created_at': pd.Timestamp.now().isoformat()
        }
    }

    model_path = "outputs/model.joblib"
    joblib.dump(artifact, model_path, compress=0)
    mlflow.log_artifact(model_path)
    
    logger.info("Job Complete.")
    mlflow.end_run()

if __name__ == "__main__":
    main()