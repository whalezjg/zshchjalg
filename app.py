"""
app.py —— 训练 / 预测 / 相关性分析 / 寻优 一体化 Web 服务

启动:
    python app.py            # 默认 0.0.0.0:5000

接口（对外字段均为英文，含义注释与接口文档说明）:
    POST /predict      预测粒径+能耗。入参: model_id + features{英文key:value}
    POST /train        训练(仅 SVR)。入参: data_path
    POST /correlation  相关性分析。入参: data_path + params[英文key]
    POST /optsearch    工艺参数寻优。入参: model_id/algorithm/alpha/... + params[]
    GET  /models       列出已训练模型

统一响应: {"code":0,"message":"success","data":{...}}
    code: 0=成功(寻优接口成功为200)，-1=失败
"""
import os
import json
import time
from flask import Flask, request, jsonify, g

from services.common import WEIGHT_ROOT, latest_dataset_id, MODEL_KEYS
from services.trainer import train
from services.predictor import predict
from services.correlation import correlation_analysis
from services.optimizer import optimize
from services.log_util import setup_logging, brief

log = setup_logging()

app = Flask(__name__)


# ============================================================
# 请求/响应日志：记录入参、耗时、返回结果
# ============================================================
@app.before_request
def _log_request_start():
    g.req_start = time.perf_counter()
    g.req_id = f"{int(time.time() * 1000) % 1000000:06d}"
    query = request.query_string.decode("utf-8", "replace")
    body = request.get_data(as_text=True)
    log.info("[%s] --> 请求 %s %s | ip=%s | query=%s | body=%s",
             g.req_id, request.method, request.path, request.remote_addr,
             query or "-", brief(body))


@app.after_request
def _log_request_end(response):
    cost_ms = (time.perf_counter() - getattr(g, "req_start", time.perf_counter())) * 1000
    try:
        body = response.get_data(as_text=True)
    except Exception:
        body = "<binary response>"
    req_id = getattr(g, "req_id", "-")
    log.info("[%s] <-- 响应 %s %s | status=%s | 耗时=%.1fms | body=%s",
             req_id, request.method, request.path, response.status_code, cost_ms, brief(body))
    return response


@app.teardown_request
def _log_request_exception(exc):
    if exc is not None:
        log.error("[%s] 未捕获异常 %s %s: %s",
                  getattr(g, "req_id", "-"), request.method, request.path, exc,
                  exc_info=(type(exc), exc, exc.__traceback__))


def _ok(data, code=0):
    return jsonify({"code": code, "message": "success", "data": data})


def _err(msg, code=-1):
    return jsonify({"code": code, "message": str(msg), "data": None})


def _get_json():
    data = request.get_json(silent=True)
    if data is None:
        raise ValueError("request body must be JSON")
    return data


def _get_feature_map(body):
    """兼容两种传参：{"features":{英文key:value}} 或平铺(除model_id外的键)"""
    if isinstance(body.get("features"), dict):
        return body["features"]
    return {k: v for k, v in body.items() if k != "model_id"}


def _resolve_data_source(body):
    """
    统一从数据库取数：忽略入参中的 data_path（前端仍传路径，此处不校验其是否存在）。
    优先级:
        1. body.dataset_id
        2. 环境变量 DEFAULT_DATASET_ID
        3. 表中最大的 dataset_id
    返回 int(dataset_id)。
    """
    dataset_id = body.get("dataset_id")
    if dataset_id is not None and str(dataset_id).strip() != "":
        try:
            return int(dataset_id)
        except (TypeError, ValueError):
            raise ValueError(f"invalid dataset_id: {dataset_id!r}")

    default_id = os.getenv("DEFAULT_DATASET_ID", "").strip()
    if default_id:
        if not default_id.isdigit():
            raise ValueError(f"invalid DEFAULT_DATASET_ID: {default_id!r}")
        return int(default_id)

    return latest_dataset_id()


# ============================================================
# 1. 预测接口
#   features 字段用英文 key（见 services/common.py PARAM_SCHEMA），
#   例如 sic3203(喷雾风机变频) / solid_content(胶体固含量) / viscosity(黏度)
# ============================================================
@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        body = _get_json()
        model_id = body.get("model_id")
        if not model_id:
            raise ValueError("missing model_id")
        features = _get_feature_map(body)
        if not features:
            raise ValueError("missing features")
        result = predict(model_id, features)
        return _ok(result)
    except Exception as e:
        return _err(e)


# ============================================================
# 2. 训练接口
#   数据固定取自数据库：可不传任何参数（默认最新 dataset_id）；
#   可选传 dataset_id 指定数据集；data_path 已废弃，传了也会忽略
#   返回: model_id/train_samples/test_samples/particle_size/energy_consumption
# ============================================================
@app.route("/train", methods=["POST"])
def api_train():
    try:
        body = request.get_json(silent=True) or {}
        source = _resolve_data_source(body)
        result = train(source)
        return _ok(result)
    except Exception as e:
        return _err(e)


# ============================================================
# 3. 相关性分析接口
#   数据固定取自数据库（同上）；params 不传时默认分析全部 10 个模型特征
#   返回: columns(英文key)/matrix(相关系数矩阵)/notes(解析说明)
# ============================================================
@app.route("/correlation", methods=["POST"])
def api_correlation():
    try:
        body = request.get_json(silent=True) or {}
        source = _resolve_data_source(body)
        params = body.get("params")
        if not params or not isinstance(params, list):
            params = list(MODEL_KEYS)
        result = correlation_analysis(source, params)
        return _ok(result)
    except Exception as e:
        return _err(e)


# ============================================================
# 4. 寻优接口
#   model_id: 模型ID(缺省取最新)  target_size: 目标粒径
#   algorithm: weighted|mopso    alpha: 粒径占比
#   params: [{name:英文key, locked:bool, value?, lower?, upper?}]
#   返回: 推荐参数组合 optimized_parameters + 预测粒径/能耗 (+mopso前沿)
# ============================================================
@app.route("/optsearch", methods=["POST"])
def api_optsearch():
    try:
        body = _get_json()

        model_id = body.get("model_id") or _latest_model_id()
        if not model_id:
            raise ValueError("no trained model, please call /train first")

        target_size = float(body.get("target_size"))
        algorithm = body.get("algorithm", "weighted")
        alpha = float(body.get("alpha", 0.7))
        max_iter = int(body.get("max_iter", 200))
        num_particles = int(body.get("num_particles", 100))

        param_configs = body.get("params")
        if not param_configs or not isinstance(param_configs, list):
            raise ValueError("missing params (list of parameter configs)")

        result = optimize(
            model_id=model_id,
            algorithm=algorithm,
            alpha=alpha,
            target_size=target_size,
            max_iter=max_iter,
            num_particles=num_particles,
            param_configs=param_configs,
        )
        return _ok(result, code=200)
    except Exception as e:
        return _err(e, code=-1)


def _latest_model_id():
    """weight/ 下最新（最大时间戳）的 model_id"""
    if not os.path.isdir(WEIGHT_ROOT):
        return None
    ids = sorted(d for d in os.listdir(WEIGHT_ROOT)
                 if os.path.isdir(os.path.join(WEIGHT_ROOT, d)))
    return ids[-1] if ids else None


# ============================================================
# 辅助: 模型列表（供前端选择 model_id）
# ============================================================
@app.route("/models", methods=["GET"])
def api_models():
    try:
        if not os.path.isdir(WEIGHT_ROOT):
            return _ok({"models": []})
        runs = []
        for mid in sorted(os.listdir(WEIGHT_ROOT)):
            cfg_path = os.path.join(WEIGHT_ROOT, mid, "config.json")
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    runs.append({
                        "model_id": mid,
                        "metrics": cfg.get("metrics", {}),
                        "data_path": cfg.get("data_path"),
                    })
                except Exception:
                    continue
        return _ok({"models": runs})
    except Exception as e:
        return _err(e)


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("应用启动 | host=0.0.0.0 port=5008 | 模型目录=%s", WEIGHT_ROOT)
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=5008, debug=False)
