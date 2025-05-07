import os,traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import uniform, randint

# --- Config ---
FILE_STATS = 'Report/OUTPUT/OUTPUT_PART1/results.csv'
FILE_VAL = 'Report/OUTPUT/OUTPUT_PART4/estimation_data_with_highest_etv.csv'
FILE_MODEL = 'Report/OUTPUT/OUTPUT_PART4/player_value_model.joblib'
FILE_PREP = 'Report/OUTPUT/OUTPUT_PART4/player_value_preprocessor.joblib'
DIR_PLOTS = 'Report/OUTPUT/OUTPUT_PART4/plots'
COL_TARGET = 'TransferValue_EUR_Millions'
ID_COLS = ['Player', 'Nation', 'Team', 'Team_TransferSite']
CAT_COLS_BASE = ['Position']

def load_data(stats_path, val_path, target):
    try:
        stats, val = pd.read_csv(stats_path), pd.read_csv(val_path)
        col_target = next((c for c in val.columns if target.lower() in c.lower()), None)
        if not col_target: return None, None, None

        val_cols = ['Player', 'Age', 'Position', col_target]
        if 'Highest_ETV' in val.columns: val_cols.append('Highest_ETV')
        val = val[[c for c in val_cols if c in val.columns]]

        if 'Player' not in stats.columns or 'Player' not in val.columns:
            return None, None, None

        df = pd.merge(stats, val, on='Player', how='left')
        for col in ['Age', 'Position']:
            for suffix in ['_value', '_stats']:
                if f'{col}{suffix}' in df.columns:
                    df[col] = df.pop(f'{col}{suffix}')
        df = df[df[col_target].notna() & (df[col_target] > 0)]
        
        return df, col_target, 'Highest_ETV' if 'Highest_ETV' in val.columns else None
    except Exception:
        traceback.print_exc()
        return None, None, None

def split_and_prep_data(df, id_cols, cat_base, col_target, col_etv):
    df[f'{col_target}_log'] = np.log1p(df[col_target])
    y = df[f'{col_target}_log']
    X = df.drop(columns=[col_target, f'{col_target}_log'] + [c for c in id_cols if c in df.columns])
    X.replace(['N/a'], np.nan, inplace=True)

    for col in X.select_dtypes('object'):
        try:
            cleaned = pd.to_numeric(X[col].astype(str).str.replace(',', ''), errors='coerce')
            if cleaned.notna().sum() > 0.8 * X[col].notna().sum(): X[col] = cleaned
        except: continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_cols = X_train.select_dtypes('number').columns.tolist()
    cat_cols = list(set(cat_base).intersection(X_train.columns).union(X_train.select_dtypes('object').columns))

    if col_etv in num_cols and col_etv in cat_cols:
        cat_cols.remove(col_etv)

    drop_cols = X_train.columns[X_train.isnull().mean() > 0.5].tolist()
    X_train.drop(columns=drop_cols, inplace=True, errors='ignore')
    X_test.drop(columns=drop_cols, inplace=True, errors='ignore')
    num_cols = [c for c in num_cols if c not in drop_cols]
    cat_cols = [c for c in cat_cols if c not in drop_cols]

    for col in cat_cols:
        X_train[col] = X_train[col].fillna('Missing').astype(str)
        X_test[col] = X_test[col].fillna('Missing').astype(str)

    return X_train, X_test, y_train, y_test, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ('num', Pipeline([('imp', KNNImputer()), ('scale', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

def evaluate(model, X_test, y_test, feat_names, out_dir):
    y_pred_log = model.predict(X_test)
    y_pred, y_true = np.expm1(y_pred_log), np.expm1(y_test)
    y_pred[y_pred < 0] = 0
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}, MAE: {mean_absolute_error(y_true, y_pred):.2f}, R2: {r2_score(y_true, y_pred):.3f}")

    if hasattr(model, 'feature_importances_') and feat_names is not None:
        fi_df = pd.DataFrame({'feature': feat_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(20)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', hue='feature', data=fi_df, palette='viridis', dodge=False, legend=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'feature_importance.png'))
        plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lim = max(y_true.max(), y_pred.max()) * 1.05
    plt.plot([0, lim], [0, lim], 'r--')
    plt.xlabel('Actual'), plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pred_vs_actual.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_true - y_pred, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'residuals.png'))
    plt.close()

def train(X_train, X_test, y_train, y_test, prep, plot_dir):
    X_train_prep, X_test_prep = prep.fit_transform(X_train), prep.transform(X_test)
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 6),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.7, 0.3)
    }
    search = RandomizedSearchCV(model, param_grid, n_iter=50, cv=5,
                                scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train_prep, y_train)
    evaluate(search.best_estimator_, X_test_prep, y_test, prep.get_feature_names_out(), plot_dir)
    return search.best_estimator_, prep

if __name__ == '__main__':
    df, col_target, col_etv = load_data(FILE_STATS, FILE_VAL, COL_TARGET)
    if df is not None:
        X_train, X_test, y_train, y_test, num_cols, cat_cols = split_and_prep_data(df, ID_COLS, CAT_COLS_BASE, col_target, col_etv)
        prep = build_preprocessor(num_cols, cat_cols)
        model, prep = train(X_train, X_test, y_train, y_test, prep, DIR_PLOTS)
