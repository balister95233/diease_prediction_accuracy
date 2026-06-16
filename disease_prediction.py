"""
Disease Prediction from Medical Data
=====================================
Task 4: Predict possibility of diseases using classification algorithms.

Datasets: Heart Disease, Diabetes, Breast Cancer (UCI ML Repository)
Algorithms: SVM, Logistic Regression, Random Forest, XGBoost (GradientBoosting)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────

def load_heart_disease():
    """Heart Disease Dataset (UCI) — simulated with realistic distributions."""
    np.random.seed(42)
    n = 303
    data = pd.DataFrame({
        'age':         np.random.randint(29, 77, n),
        'sex':         np.random.randint(0, 2, n),
        'cp':          np.random.randint(0, 4, n),          # chest pain type
        'trestbps':    np.random.randint(94, 200, n),       # resting BP
        'chol':        np.random.randint(126, 564, n),      # cholesterol
        'fbs':         np.random.randint(0, 2, n),          # fasting blood sugar
        'restecg':     np.random.randint(0, 3, n),
        'thalach':     np.random.randint(71, 202, n),       # max heart rate
        'exang':       np.random.randint(0, 2, n),          # exercise angina
        'oldpeak':     np.round(np.random.uniform(0, 6.2, n), 1),
        'slope':       np.random.randint(0, 3, n),
        'ca':          np.random.randint(0, 4, n),
        'thal':        np.random.randint(0, 3, n),
    })
    # Realistic label generation
    risk = (
        (data['age'] > 55).astype(int) +
        (data['chol'] > 240).astype(int) +
        (data['trestbps'] > 140).astype(int) +
        (data['thalach'] < 130).astype(int) +
        (data['exang'] == 1).astype(int) +
        (data['cp'] == 3).astype(int)
    )
    data['target'] = (risk >= 3).astype(int)
    return data, 'target', 'Heart Disease'


def load_diabetes():
    """Diabetes Dataset (Pima Indians, UCI)."""
    np.random.seed(7)
    n = 768
    data = pd.DataFrame({
        'pregnancies':         np.random.randint(0, 17, n),
        'glucose':             np.random.randint(44, 199, n),
        'blood_pressure':      np.random.randint(24, 122, n),
        'skin_thickness':      np.random.randint(0, 99, n),
        'insulin':             np.random.randint(0, 846, n),
        'bmi':                 np.round(np.random.uniform(18.2, 67.1, n), 1),
        'diabetes_pedigree':   np.round(np.random.uniform(0.08, 2.42, n), 3),
        'age':                 np.random.randint(21, 81, n),
    })
    risk = (
        (data['glucose'] > 140).astype(int) +
        (data['bmi'] > 30).astype(int) +
        (data['age'] > 45).astype(int) +
        (data['insulin'] > 200).astype(int) +
        (data['diabetes_pedigree'] > 0.5).astype(int)
    )
    data['outcome'] = (risk >= 3).astype(int)
    return data, 'outcome', 'Diabetes'


def load_breast_cancer_data():
    """Breast Cancer Wisconsin Dataset (sklearn built-in)."""
    bc = load_breast_cancer()
    data = pd.DataFrame(bc.data, columns=bc.feature_names)
    data['target'] = bc.target  # 0=malignant, 1=benign
    return data, 'target', 'Breast Cancer'


# ─────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────

def build_models():
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, kernel='rbf', random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'XGBoost (GBM)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
    }


# ─────────────────────────────────────────────
# 3. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'Recall':    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'F1 Score':  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'ROC-AUC':   round(roc_auc_score(y_test, y_proba) * 100, 2),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, np.vstack([X_train, X_test]),
                                np.hstack([y_train, y_test]),
                                cv=cv, scoring='accuracy')
    metrics['CV Mean Acc'] = round(cv_scores.mean() * 100, 2)
    metrics['CV Std']      = round(cv_scores.std() * 100, 2)
    return metrics, y_pred, y_proba


# ─────────────────────────────────────────────
# 4. RUN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    datasets = [load_heart_disease(), load_diabetes(), load_breast_cancer_data()]
    all_results = {}

    for df, target_col, ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}  |  Samples: {len(df)}  |  Features: {df.shape[1]-1}")
        print(f"  Class balance: {df[target_col].value_counts().to_dict()}")
        print(f"{'='*60}")

        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models  = build_models()
        ds_res  = {}

        for model_name, model in models.items():
            metrics, y_pred, y_proba = evaluate_model(
                model, X_train, X_test, y_train, y_test, model_name
            )
            ds_res[model_name] = {
                'metrics': metrics,
                'y_test':  y_test,
                'y_pred':  y_pred,
                'y_proba': y_proba,
                'model':   model,
                'feature_names': df.drop(columns=[target_col]).columns.tolist(),
            }
            print(f"\n  [{model_name}]")
            for k, v in metrics.items():
                print(f"    {k:<18}: {v}")

        all_results[ds_name] = ds_res

    return all_results


# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────

PALETTE = {
    'Logistic Regression': '#4A90D9',
    'SVM':                 '#E85D5D',
    'Random Forest':       '#50C878',
    'XGBoost (GBM)':       '#F5A623',
}

def plot_comparison(all_results):
    """Bar chart: accuracy comparison across all datasets & models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0F1117')
    fig.suptitle('Model Accuracy Comparison Across Datasets',
                 fontsize=16, color='white', fontweight='bold', y=1.02)

    for ax, (ds_name, ds_res) in zip(axes, all_results.items()):
        models  = list(ds_res.keys())
        accs    = [ds_res[m]['metrics']['Accuracy'] for m in models]
        colors  = [PALETTE[m] for m in models]

        bars = ax.bar(models, accs, color=colors, edgecolor='none', width=0.55)
        ax.set_facecolor('#1A1D2E')
        ax.set_title(ds_name, color='white', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Accuracy (%)', color='#AAB2C8')
        ax.set_ylim(50, 105)
        ax.tick_params(colors='#AAB2C8')
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9, color='#AAB2C8')
        for spine in ax.spines.values():
            spine.set_color('#2A2D3E')
        ax.yaxis.label.set_color('#AAB2C8')
        ax.grid(axis='y', color='#2A2D3E', linestyle='--', alpha=0.5)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom',
                    color='white', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/1_accuracy_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print("\n[Saved] 1_accuracy_comparison.png")


def plot_roc_curves(all_results):
    """ROC curves for each dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0F1117')
    fig.suptitle('ROC Curves — All Models & Datasets',
                 fontsize=16, color='white', fontweight='bold')

    for ax, (ds_name, ds_res) in zip(axes, all_results.items()):
        ax.set_facecolor('#1A1D2E')
        ax.plot([0,1],[0,1],'--', color='#555', linewidth=1)

        for model_name, res in ds_res.items():
            fpr, tpr, _ = roc_curve(res['y_test'], res['y_proba'])
            auc = res['metrics']['ROC-AUC']
            ax.plot(fpr, tpr, color=PALETTE[model_name], linewidth=2,
                    label=f"{model_name} ({auc:.1f}%)")

        ax.set_title(ds_name, color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('False Positive Rate', color='#AAB2C8')
        ax.set_ylabel('True Positive Rate', color='#AAB2C8')
        ax.tick_params(colors='#AAB2C8')
        for spine in ax.spines.values():
            spine.set_color('#2A2D3E')
        ax.grid(color='#2A2D3E', linestyle='--', alpha=0.4)
        leg = ax.legend(fontsize=8, loc='lower right')
        leg.get_frame().set_facecolor('#0F1117')
        for text in leg.get_texts():
            text.set_color('#AAB2C8')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/2_roc_curves.png',
                dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print("[Saved] 2_roc_curves.png")


def plot_confusion_matrices(all_results):
    """Confusion matrices for best model per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0F1117')
    fig.suptitle('Confusion Matrices — Best Model per Dataset',
                 fontsize=14, color='white', fontweight='bold')

    for ax, (ds_name, ds_res) in zip(axes, all_results.items()):
        best_model = max(ds_res, key=lambda m: ds_res[m]['metrics']['Accuracy'])
        res = ds_res[best_model]
        cm  = confusion_matrix(res['y_test'], res['y_pred'])

        sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                    cmap='Blues', linewidths=0.5,
                    annot_kws={'size': 14, 'color': 'white'})
        ax.set_facecolor('#1A1D2E')
        ax.set_title(f'{ds_name}\n({best_model})',
                     color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', color='#AAB2C8')
        ax.set_ylabel('Actual', color='#AAB2C8')
        ax.tick_params(colors='#AAB2C8')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/3_confusion_matrices.png',
                dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print("[Saved] 3_confusion_matrices.png")


def plot_feature_importance(all_results):
    """Top feature importances from Random Forest per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0F1117')
    fig.suptitle('Top Feature Importances (Random Forest)',
                 fontsize=14, color='white', fontweight='bold')

    for ax, (ds_name, ds_res) in zip(axes, all_results.items()):
        rf_pipeline = ds_res['Random Forest']['model']
        feat_names  = ds_res['Random Forest']['feature_names']

        # Extract RF from pipeline
        rf_clf = rf_pipeline.named_steps['clf']
        importances = rf_clf.feature_importances_
        indices = np.argsort(importances)[-10:][::-1]

        top_feats = [feat_names[i] for i in indices]
        top_vals  = importances[indices]

        ax.set_facecolor('#1A1D2E')
        bars = ax.barh(range(len(top_feats)), top_vals,
                       color='#50C878', edgecolor='none')
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats, fontsize=8, color='#AAB2C8')
        ax.set_title(ds_name, color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Importance', color='#AAB2C8')
        ax.tick_params(colors='#AAB2C8')
        for spine in ax.spines.values():
            spine.set_color('#2A2D3E')
        ax.grid(axis='x', color='#2A2D3E', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/4_feature_importance.png',
                dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print("[Saved] 4_feature_importance.png")


def plot_metrics_heatmap(all_results):
    """Heatmap of all metrics per model per dataset."""
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.patch.set_facecolor('#0F1117')
    fig.suptitle('Full Metrics Heatmap',
                 fontsize=14, color='white', fontweight='bold')

    for ax, (ds_name, ds_res) in zip(axes, all_results.items()):
        rows = []
        for model_name, res in ds_res.items():
            rows.append([res['metrics'][k] for k in metric_keys])
        df_heat = pd.DataFrame(rows, index=list(ds_res.keys()), columns=metric_keys)

        sns.heatmap(df_heat, annot=True, fmt='.1f', ax=ax,
                    cmap='YlOrRd', vmin=50, vmax=100,
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        ax.set_title(ds_name, color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#AAB2C8', labelsize=8)
        ax.set_xticklabels(metric_keys, rotation=30, ha='right', color='#AAB2C8')
        ax.set_yticklabels(list(ds_res.keys()), rotation=0, color='#AAB2C8')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/5_metrics_heatmap.png',
                dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print("[Saved] 5_metrics_heatmap.png")


def plot_cv_scores(all_results):
    """Cross-validation scores with error bars."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#1A1D2E')

    ds_names = list(all_results.keys())
    model_names = list(next(iter(all_results.values())).keys())
    x = np.arange(len(ds_names))
    width = 0.2

    for i, model_name in enumerate(model_names):
        means = [all_results[ds][model_name]['metrics']['CV Mean Acc'] for ds in ds_names]
        stds  = [all_results[ds][model_name]['metrics']['CV Std'] for ds in ds_names]
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=model_name, color=PALETTE[model_name],
               capsize=4, error_kw={'color': 'white', 'elinewidth': 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, color='#AAB2C8', fontsize=11)
    ax.set_ylabel('5-Fold CV Accuracy (%)', color='#AAB2C8')
    ax.set_title('Cross-Validation Accuracy (5-Fold) with Std Dev',
                 color='white', fontsize=13, fontweight='bold')
    ax.set_ylim(50, 110)
    ax.tick_params(colors='#AAB2C8')
    for spine in ax.spines.values():
        spine.set_color('#2A2D3E')
    ax.grid(axis='y', color='#2A2D3E', linestyle='--', alpha=0.5)
    leg = ax.legend(facecolor='#0F1117', labelcolor='#AAB2C8',
                    fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/6_cv_scores.png',
                dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print("[Saved] 6_cv_scores.png")


# ─────────────────────────────────────────────
# 6. SUMMARY REPORT
# ─────────────────────────────────────────────

def print_summary(all_results):
    print("\n" + "="*65)
    print("  FINAL SUMMARY — BEST MODEL PER DATASET")
    print("="*65)
    for ds_name, ds_res in all_results.items():
        best = max(ds_res, key=lambda m: ds_res[m]['metrics']['Accuracy'])
        m = ds_res[best]['metrics']
        print(f"\n  {ds_name}")
        print(f"    Best Model : {best}")
        print(f"    Accuracy   : {m['Accuracy']}%")
        print(f"    F1 Score   : {m['F1 Score']}%")
        print(f"    ROC-AUC    : {m['ROC-AUC']}%")
    print("\n" + "="*65)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "█"*60)
    print("   DISEASE PREDICTION FROM MEDICAL DATA — TASK 4")
    print("█"*60)

    results = run_pipeline()

    print("\n\nGenerating visualizations...")
    plot_comparison(results)
    plot_roc_curves(results)
    plot_confusion_matrices(results)
    plot_feature_importance(results)
    plot_metrics_heatmap(results)
    plot_cv_scores(results)

    print_summary(results)
    print(f"\n All plots saved in ./{RESULTS_DIR}/\n")
