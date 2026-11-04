"""
trainer.py —— 训练核心

流程（沿用原 train 逻辑）:
    1. 读取数据集（.csv/.xlsx），按 PARAM_SCHEMA 自动解析特征/目标列
    2. train_test_split(test_size=0.2, random_state=34)，粒径/能耗同一次划分
    3. StandardScaler 分别标准化 X、粒径、能耗
    4. SVR 网格搜索(C/kernel) + 5折交叉验证(neg_mse) 选参
    5. 用最优参数在全量训练集重新拟合
    6. 测试集反标准化后计算 MSE / MAPE(%) / 1-MAPE / R2
    7. 保存到 weight/<model_id>/，返回英文 key 指标

对外字段均为英文：
    model_id / train_samples / test_samples
    particle_size:  {mse, mape, 1_mape, r2}
    energy_consumption: {mse, mape, 1_mape, r2}
"""
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

from services.common import (
    model_dir, gen_model_id, read_dataframe,
    resolve_model_feature_columns, resolve_target_columns,
    mean_absolute_percentage_error, save_pickle, save_json,
    RANDOM_SEED, TEST_SIZE, SVR_PARAM_GRID, TARGET_MODEL_FILES,
)
from services.log_util import get_logger

log = get_logger("trainer")


def _grid_search_svr(X_train, y_train):
    """SVR 网格搜索: C/kernel + 5折CV(neg_mse)"""
    best_score, best_params = float("inf"), {}
    keys, values = zip(*SVR_PARAM_GRID.items())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        model = SVR(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_squared_error",
        )
        mse_cv = -np.mean(scores)
        if mse_cv < best_score:
            best_score, best_params = mse_cv, params
    return best_params, best_score


def _evaluate(y_true_real, y_pred_real):
    mse = float(mean_squared_error(y_true_real, y_pred_real))
    mape = float(mean_absolute_percentage_error(y_true_real, y_pred_real))
    r2 = float(r2_score(y_true_real, y_pred_real))
    return {
        "mse": round(mse, 4),
        "mape": round(mape, 4),
        "1_mape": round(1 - mape / 100, 4),
        "r2": round(r2, 4),
    }


def train(data_path):
    """
    训练粒径/能耗 + 粒度分布(5 档) SVR 模型

    SVR 为单输出，因此每个目标列（粒径、能耗、5 档粒度分布）各训练一个 SVR；
    特征共享同一次标准化与切分，模型/scaler 按 TARGET_MODEL_FILES 命名保存。

    返回:
        dict: model_id, train_samples, test_samples,
              particle_size{...}, energy_consumption{...},
              psd_0_20{...} ... psd_0_149{...}
    """
    # ---------- 1. 读取 ----------
    log.info("训练开始 | data_path=%s", data_path)
    df = read_dataframe(data_path)
    feature_cols = resolve_model_feature_columns(df)
    target_cols = resolve_target_columns(df)  # {英文key: 数据集列名}，含粒度分布
    log.info("读取数据完成 | 行数=%d | 特征=%s | 目标=%s",
             len(df), feature_cols, list(target_cols.keys()))

    needed = feature_cols + list(target_cols.values())
    clean = df[needed].copy()
    for c in needed:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna(subset=needed).reset_index(drop=True)
    if len(clean) < 10:
        raise ValueError(f"clean samples too few: {len(clean)}")

    X = clean[feature_cols].values.astype(float)
    y_all = {k: clean[col].values.astype(float) for k, col in target_cols.items()}

    # ---------- 2. 划分（所有目标使用同一切分索引）----------
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    log.info("数据划分完成 | 训练=%d 测试=%d", len(train_idx), len(test_idx))

    # ---------- 3. 特征标准化 ----------
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw)
    X_test = scaler_X.transform(X_test_raw)

    # ---------- 4. 逐目标训练 + 评估 + 保存 ----------
    model_id = gen_model_id()
    run_dir = model_dir(model_id)
    os.makedirs(run_dir, exist_ok=True)
    save_pickle(scaler_X, os.path.join(run_dir, "scaler_X.pkl"))

    metrics = {}
    svr_params = {}
    for key in list(target_cols.keys()):
        y_tr = y_all[key][train_idx].reshape(-1, 1)
        y_te = y_all[key][test_idx].reshape(-1, 1)

        scaler = StandardScaler()
        y_tr_n = scaler.fit_transform(y_tr).ravel()

        params, _ = _grid_search_svr(X_train, y_tr_n)
        model = SVR(**params)
        model.fit(X_train, y_tr_n)

        y_pred_n = model.predict(X_test)
        y_te_real = y_te.ravel()
        y_pred_real = scaler.inverse_transform(y_pred_n.reshape(-1, 1)).ravel()
        metrics[key] = _evaluate(y_te_real, y_pred_real)
        svr_params[key] = params

        model_file, scaler_file = TARGET_MODEL_FILES[key]
        save_pickle(model, os.path.join(run_dir, model_file))
        save_pickle(scaler, os.path.join(run_dir, scaler_file))
        log.info("训练目标=%s 完成 | 参数=%s | 指标=%s", key, params, metrics[key])

    # ---------- 5. 特征顺序/边界/元信息 ----------
    save_json(feature_cols, os.path.join(run_dir, "feature_columns.json"))
    save_json({
        str(col): [float(np.min(clean[col])), float(np.max(clean[col]))]
        for col in feature_cols
    }, os.path.join(run_dir, "feature_bounds.json"))

    meta = {
        "model_id": model_id,
        "data_path": data_path,
        "feature_key_order": [p["key"] for p in _model_params_schema()],
        "targets": target_cols,          # 数据集目标列名
        "train_samples": int(X_train_raw.shape[0]),
        "test_samples": int(X_test_raw.shape[0]),
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "svr_params": svr_params,
        "metrics": metrics,
    }
    save_json(meta, os.path.join(run_dir, "config.json"))

    result = {
        "model_id": model_id,
        "train_samples": meta["train_samples"],
        "test_samples": meta["test_samples"],
    }
    result.update(metrics)
    log.info("训练结束 | model_id=%s | 已保存到 %s", model_id, run_dir)
    return result


def _model_params_schema():
    from services.common import MODEL_PARAMS
    return MODEL_PARAMS
