"""
predictor.py —— 预测核心

根据 model_id 加载 weight/<model_id>/ 下的粒径/能耗 SVR 与 scaler，
按训练时保存的特征列顺序构造输入向量。

对外入参使用英文 key（见 common.PARAM_SCHEMA），例如：
    sic3203 -> 喷雾风机变频 SIC3203(Hz)
    solid_content -> 胶体固含量
    gun_count -> 喷枪数量
    viscosity -> 黏度
展示参数 nozzle_31_count / nozzle_33_count 请求中可携带，服务端自动忽略。
"""
import os
import numpy as np

from services.common import (
    model_dir, load_pickle, load_json,
    PSD_KEYS, TARGET_MODEL_FILES,
)
from services.log_util import get_logger

log = get_logger("predictor")


class RunBundle:
    """一次训练（model_id）对应的全部推理产物"""

    def __init__(self, model_id):
        run_dir = model_dir(model_id)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"model dir not found: {run_dir} (model_id={model_id})")

        self.model_id = model_id
        self.run_dir = run_dir
        self.config = load_json(os.path.join(run_dir, "config.json"))
        # 特征顺序：feature_columns.json = [数据集列名...]，与英文 key 顺序一一对应
        self.feature_columns = self.config.get("feature_columns") or load_json(
            os.path.join(run_dir, "feature_columns.json"))
        self.feature_key_order = self.config.get("feature_key_order")
        self.feature_bounds = load_json(os.path.join(run_dir, "feature_bounds.json"))
        self.particle_model = load_pickle(os.path.join(run_dir, "particle_svr.pkl"))
        self.energy_model = load_pickle(os.path.join(run_dir, "energy_svr.pkl"))
        self.scaler_X = load_pickle(os.path.join(run_dir, "scaler_X.pkl"))
        self.scaler_y = load_pickle(os.path.join(run_dir, "scaler_y.pkl"))
        self.scaler_energy = load_pickle(os.path.join(run_dir, "scaler_energy.pkl"))

        # 粒度分布模型/scaler（可选：旧模型无这些文件时置空，不影响粒径/能耗）
        self.psd_models = {}
        self.psd_scalers = {}
        for key in PSD_KEYS:
            model_file, scaler_file = TARGET_MODEL_FILES[key]
            mp = os.path.join(run_dir, model_file)
            sp = os.path.join(run_dir, scaler_file)
            if os.path.isfile(mp) and os.path.isfile(sp):
                self.psd_models[key] = load_pickle(mp)
                self.psd_scalers[key] = load_pickle(sp)


def build_feature_vector(bundle, feature_map):
    """
    按模型特征顺序构造 1xN 原始尺度特征向量

    feature_map: {英文key: 数值}（常见展示参数如 nozzle_31_count 自动忽略）
    """
    row = []
    for i, col in enumerate(bundle.feature_columns):
        # 英文 key 顺序与特征列顺序一一对应（由训练时 feature_key_order 记录）
        key = bundle.feature_key_order[i] if bundle.feature_key_order else col
        if key not in feature_map:
            raise ValueError(
                f"missing model feature: '{key}' (对应数据集列 {col})"
            )
        row.append(float(feature_map[key]))
    if len(row) != len(bundle.feature_columns):
        raise ValueError(f"feature count mismatch: expect {len(bundle.feature_columns)}")
    return np.array([row], dtype=float)


def predict(model_id, feature_map):
    """
    执行粒径+能耗+粒度分布预测

    返回:
        model_id, particle_size_um, energy_consumption,
        以及 psd_0_20 ... psd_0_149（旧模型无粒度分布时这些字段省略）
    """
    bundle = RunBundle(model_id)
    X = build_feature_vector(bundle, feature_map)
    X_scaled = bundle.scaler_X.transform(X)

    y_norm = bundle.particle_model.predict(X_scaled)
    particle = float(bundle.scaler_y.inverse_transform(y_norm.reshape(-1, 1)).ravel()[0])

    en_norm = bundle.energy_model.predict(X_scaled)
    energy = float(bundle.scaler_energy.inverse_transform(en_norm.reshape(-1, 1)).ravel()[0])

    result = {
        "model_id": model_id,
        "particle_size_um": round(particle, 4),
        "energy_consumption": round(energy, 4),
    }

    for key in PSD_KEYS:
        model = bundle.psd_models.get(key)
        scaler = bundle.psd_scalers.get(key)
        if model is not None and scaler is not None:
            p = model.predict(X_scaled)
            val = float(scaler.inverse_transform(p.reshape(-1, 1)).ravel()[0])
            result[key] = round(val, 4)

    log.info("预测完成 | model_id=%s | 入参=%s | 结果=%s", model_id, feature_map, result)
    return result
