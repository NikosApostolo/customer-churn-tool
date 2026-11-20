"""Small, fixed nested-validation experiment; never uses the published test set."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import clean_data
from src.model import make_preprocessor

def stability(source: Path) -> None:
    """Follow-up stability check; these folds overlap, so they are not independent."""
    X, y, _, _ = clean_data(pd.read_csv(source))
    dev, _ = train_test_split(np.arange(len(X)), test_size=.2, stratify=y, random_state=42)
    X, y = X.iloc[dev].reset_index(drop=True), y.iloc[dev].reset_index(drop=True)
    rows = []
    configs = {
        'current_xgb': dict(n_estimators=180, max_depth=3, learning_rate=.05, reg_lambda=2),
        'shallow_xgb': dict(n_estimators=400, max_depth=2, learning_rate=.03, reg_lambda=5, min_child_weight=10),
    }
    for seed in [2027, 2028]:
        for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y), 1):
            for name, params in configs.items():
                model = XGBClassifier(**params, objective='binary:logistic', eval_metric='logloss', random_state=42,
                                      n_jobs=2, tree_method='hist', subsample=.9, colsample_bytree=.9)
                pipeline = Pipeline([('prepare', make_preprocessor(X)), ('model', model)]).fit(X.iloc[a], y.iloc[a])
                p = pipeline.predict_proba(X.iloc[b])[:, 1]
                rows.append(dict(seed=seed, fold=fold, model=name, ap=average_precision_score(y.iloc[b], p),
                                 brier=brier_score_loss(y.iloc[b], p)))
    Path(__file__).with_name('stability.json').write_text(json.dumps(rows, indent=2))
    print(pd.DataFrame(rows).groupby('model')[['ap', 'brier']].mean())


def main():
    source = Path(sys.argv[1])
    X, y, ids, _ = clean_data(pd.read_csv(source))
    development, excluded = train_test_split(np.arange(len(X)), test_size=.2, stratify=y, random_state=42)
    X, y = X.iloc[development].reset_index(drop=True), y.iloc[development].reset_index(drop=True)
    common = dict(objective='binary:logistic', eval_metric='logloss', random_state=42,
                  n_jobs=2, tree_method='hist', subsample=.9, colsample_bytree=.9)
    candidates = {
        'current_xgb': XGBClassifier(**common, n_estimators=180, max_depth=3, learning_rate=.05, reg_lambda=2),
        'shallow_xgb': XGBClassifier(**common, n_estimators=400, max_depth=2, learning_rate=.03, reg_lambda=5, min_child_weight=10),
        'regularized_xgb': XGBClassifier(**common, n_estimators=400, max_depth=3, learning_rate=.03, reg_lambda=10, min_child_weight=10),
        'logistic_C1': LogisticRegression(C=1, max_iter=2000, random_state=42),
        'logistic_C01': LogisticRegression(C=.1, max_iter=2000, random_state=42),
    }
    rows=[]
    for fold,(train,valid) in enumerate(StratifiedKFold(5,shuffle=True,random_state=2026).split(X,y),1):
        inner=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X.iloc[train],y.iloc[train]))
        scores={}
        fitted={}
        for name,estimator in candidates.items():
            ap=[]
            for a,b in inner:
                pipeline=Pipeline([('prepare',make_preprocessor(X)),('model',clone(estimator))])
                pipeline.fit(X.iloc[train[a]],y.iloc[train[a]])
                ap.append(average_precision_score(y.iloc[train[b]],pipeline.predict_proba(X.iloc[train[b]])[:,1]))
            scores[name]=float(np.mean(ap))
            fitted[name]=Pipeline([('prepare',make_preprocessor(X)),('model',clone(estimator))]).fit(X.iloc[train],y.iloc[train])
        winner=max(scores,key=scores.get)
        for name,pipeline in fitted.items():
            probability=pipeline.predict_proba(X.iloc[valid])[:,1]
            rows.append(dict(fold=fold,model=name,inner_ap=scores[name],ap=average_precision_score(y.iloc[valid],probability),brier=brier_score_loss(y.iloc[valid],probability),selected=name==winner))
        Path(__file__).with_name('results.json').write_text(json.dumps(dict(data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),development_rows=len(X),excluded_published_test_rows=len(excluded),rows=rows),indent=2))
        print('Fold',fold,'selected',winner,'AP',round(next(r['ap'] for r in rows if r['fold']==fold and r['selected']),4),flush=True)
    result=pd.DataFrame(rows)
    print(result.groupby('model')[['ap','brier']].mean().round(5).to_string())
    print('Nested selection AP:',result[result.selected].ap.mean())

if __name__=='__main__':
    if '--stability' in sys.argv:
        stability(Path(sys.argv[1]))
    else:
        main()
