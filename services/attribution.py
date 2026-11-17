"""
attribution.py —— 目标偏差归因（根因分析）

场景: 给定一套工艺参数，预测粒径与标准值（如 76μm）偏差较大，
      定位是哪些参数造成的。

偏差恒等式（不漏项）:
    y - target = (y - y_hat) + (y_hat - y_hat(x_ref)) + (y_hat(x_ref) - target)
    实测偏差      模型误差        参数可归因              基准偏移

归因算法:
    ig            积分梯度，满足完备性 sum(ig) = y_hat(x) - y_hat(x_ref)
    group_shapley 共线特征合并后的精确 Shapley（特征数 <= 12 时可用）
    oat           梯度 x 偏离，直观展示（一阶近似）

参照点 x_ref 优先级:
    1. 入参 reference.values（用户指定的标准工况）
    2. 入参 reference.product -> 数据库同产品中位数
    3. 数据库最新数据集全量中位数
    4. 训练特征边界中点（兜底，精度最差）

共线处理: 从数据库读训练数据，|相关系数| >= COLLINEAR_R 的特征合并为一组，
          组内不单独排序（否则贡献会在组内随机分配）。
"""
import itertools
import math

import numpy as np

from services.common import (
    MODEL_KEYS, TARGET_MODEL_FILES, load_dataset_from_db, latest_dataset_id,
    resolve_model_feature_columns,
)
from services.predictor import RunBundle
from services.log_util import get_logger

log = get_logger("attribution")

COLLINEAR_R = 0.95      # 共线合并阈值
MAX_GROUPS_FOR_SHAPLEY = 12   # 组数过多时 2^n 枚举不可行
CONTROLLABLE_KEYS = {
    "sic3203", "sic1014", "sic3204", "prc3101", "fi3108", "sic1008", "sic1009",
}
INT_KEYS = {"gun_count"}


# ============================================================
# 模型与梯度
# ============================================================
class _Model:
    """封装一次训练产物，提供预测与解析梯度"""

    def __init__(self, model_id):
        self.bundle = RunBundle(model_id)
        self.model_id = model_id
        self.keys = list(self.bundle.feature_key_order or MODEL_KEYS)
        self.cols = list(self.bundle.feature_columns)
        self.svr = self.bundle.particle_model
        self.scaler_X = self.bundle.scaler_X
        self.scaler_y = self.bundle.scaler_y

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = self.scaler_X.transform(X)
        p = self.svr.predict(Xs)
        return self.scaler_y.inverse_transform(p.reshape(-1, 1)).ravel()

    def predict_one(self, x):
        return float(self.predict(x)[0])

    def grad(self, x):
        """RBF 核 SVR 解析梯度，返回原始尺度 dy/dx

        support_vectors_ 已在训练时的标准化空间，不可再次 transform。
        """
        sv = self.svr.support_vectors_
        alpha = self.svr.dual_coef_[0]
        gamma = self.svr._gamma
        xs = self.scaler_X.transform(np.asarray(x, dtype=float)[None, :])
        diff = sv - xs
        K = np.exp(-gamma * np.sum(diff ** 2, axis=1))
        g = 2.0 * gamma * (alpha * K) @ diff
        return g / self.scaler_X.scale_ * self.scaler_y.scale_

    def bounds(self, key):
        col = self.cols[self.keys.index(key)]
        lo, hi = self.bundle.feature_bounds[col]
        return float(lo), float(hi)


# ============================================================
# 参照点
# ============================================================
def _midpoint_reference(m):
    vals = []
    for k in m.keys:
        lo, hi = m.bounds(k)
        vals.append((lo + hi) / 2.0)
    return np.array(vals, dtype=float)


def _find_product_col(df):
    for c in df.columns:
        name = str(c).lower()
        if "样品" in str(c) or "产品" in str(c) or "product" in name or "sample_name" in name:
            return c
    return None


def _build_reference(m, reference, dataset_id):
    """返回 (x_ref, source_info, warning)"""
    reference = reference or "auto"

    if isinstance(reference, dict) and reference.get("values"):
        vals = reference["values"]
        miss = [k for k in m.keys if k not in vals]
        if miss:
            raise ValueError(f"reference.values missing keys: {miss}")
        x_ref = np.array([float(vals[k]) for k in m.keys], dtype=float)
        return x_ref, {"source": "user_provided", "n": None}, None

    product = reference.get("product") if isinstance(reference, dict) else None

    try:
        did = int(dataset_id) if dataset_id is not None else latest_dataset_id()
        df = load_dataset_from_db(did)
        fcols = resolve_model_feature_columns(df)

        import pandas as pd
        num = df[fcols].apply(pd.to_numeric, errors="coerce").dropna()

        if product and _find_product_col(df) is not None:
            pcol = _find_product_col(df)
            mask = df.loc[num.index, pcol].astype(str).str.contains(str(product), na=False)
            if mask.sum() >= 3:
                num = num.loc[mask[mask].index]

        if len(num) < 3:
            raise ValueError(f"too few rows for reference: {len(num)}")

        x_ref = num.median().values.astype(float)
        info = {
            "source": "same_product_median" if product else "dataset_median",
            "dataset_id": did,
            "product": product,
            "n": int(len(num)),
        }
        if product and info["source"] == "dataset_median":
            info["warn_product_not_found"] = True
        return x_ref, info, None

    except Exception as e:  # 数据库不可用时降级
        x_ref = _midpoint_reference(m)
        return x_ref, {"source": "bounds_midpoint", "n": None}, \
            f"reference fallback to feature bounds midpoint: {e}"


# ============================================================
# 共线分组
# ============================================================
def _collinear_groups(m, dataset_id):
    """返回 (groups, info)。groups 为 list[list[特征下标]]，无数据时每特征一组"""
    single = [[i] for i in range(len(m.keys))]
    try:
        did = int(dataset_id) if dataset_id is not None else latest_dataset_id()
        df = load_dataset_from_db(did)
        fcols = resolve_model_feature_columns(df)
        import pandas as pd
        num = df[fcols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(num) < 10:
            return single, {"source": "none", "reason": "too few rows"}

        corr = num.corr().abs().values
        n = len(fcols)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        merged = []
        for i in range(n):
            for j in range(i + 1, n):
                if corr[i, j] >= COLLINEAR_R:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
                    merged.append({
                        "members": [m.keys[i], m.keys[j]],
                        "r": round(float(corr[i, j]), 4),
                    })

        buckets = {}
        for i in range(n):
            buckets.setdefault(find(i), []).append(i)
        groups = sorted(buckets.values(), key=lambda g: g[0])
        info = {
            "source": "dataset_correlation",
            "dataset_id": did,
            "threshold": COLLINEAR_R,
            "merged_pairs": merged,
        }
        return groups, info
    except Exception as e:
        return single, {"source": "none", "reason": str(e)}


# ============================================================
# 归因算法
# ============================================================
def _ig(m, x, x_ref, steps=64):
    """积分梯度：沿 x_ref -> x 直线路径对梯度积分"""
    trapz = getattr(np, "trapezoid", np.trapz)
    alphas = np.linspace(0.0, 1.0, int(steps) + 1)
    grads = np.array([m.grad(x_ref + a * (x - x_ref)) for a in alphas])
    avg = trapz(grads, alphas, axis=0)
    return (x - x_ref) * avg


def _oat(m, x, x_ref):
    return m.grad(x) * (x - x_ref)


def _group_shapley(m, x, x_ref, groups):
    """精确 Shapley（组级），组数 <= 12"""
    n = len(groups)
    if n > MAX_GROUPS_FOR_SHAPLEY:
        return None
    vals = np.zeros(2 ** n)
    for mask in range(2 ** n):
        v = np.array(x_ref, dtype=float)
        for gi in range(n):
            if (mask >> gi) & 1:
                for idx in groups[gi]:
                    v[idx] = x[idx]
        vals[mask] = m.predict_one(v)

    phi = np.zeros(n)
    facts = [math.factorial(k) for k in range(n + 1)]
    for gi in range(n):
        others = [k for k in range(n) if k != gi]
        for r in range(len(others) + 1):
            w = facts[r] * facts[n - r - 1] / facts[n]
            for S in itertools.combinations(others, r):
                sm = 0
                for k in S:
                    sm |= 1 << k
                phi[gi] += w * (vals[sm | (1 << gi)] - vals[sm])
    return phi


def _counterfactual(m, x, x_ref, target, top_k=3):
    """单参数二分搜索，使预测回到目标值"""
    out = []
    for j in np.argsort(-np.abs(np.asarray(x) - np.asarray(x_ref))):
        key = m.keys[j]
        lo, hi = m.bounds(key)

        def g(v):
            xx = np.array(x, dtype=float)
            xx[j] = v
            return m.predict_one(xx) - target

        a, b = lo, hi
        ga, gb = g(a), g(b)
        if ga * gb > 0:
            continue
        for _ in range(70):
            mid = (a + b) / 2.0
            if g(a) * g(mid) <= 0:
                b = mid
            else:
                a = mid
        new_val = (a + b) / 2.0
        if key in INT_KEYS:
            new_val = float(round(new_val))
        out.append({
            "name": key,
            "current": round(float(x[j]), 4),
            "suggested": round(float(new_val), 4),
            "delta": round(float(new_val - x[j]), 4),
            "expected_size": round(float(m.predict_one(np.where(
                np.arange(len(x)) == j, new_val, x))), 4),
        })
        if len(out) >= int(top_k):
            break
    return out


# ============================================================
# 对外入口
# ============================================================
def analyze(model_id, features, target_size, observed_size=None,
            reference=None, methods=None, top_k=5, ig_steps=64, dataset_id=None):
    """
    目标偏差归因

    model_id     : 模型 ID
    features     : {英文key: 值}，需包含全部模型特征
    target_size  : 标准粒径（μm）
    observed_size: 实测粒径（可选；提供后单独给出模型误差）
    reference    : "auto" | {"values": {...}} | {"product": "..."}
    methods      : ["ig","group_shapley","oat"]，默认全开
    top_k        : 返回贡献前 K 项
    dataset_id   : 用于构造参照点与共线检测的数据集，默认最新
    """
    methods = methods or ["ig", "group_shapley", "oat"]
    m = _Model(model_id)

    missing = [k for k in m.keys if k not in features]
    if missing:
        raise ValueError(f"missing model feature: {missing}")

    x = np.array([float(features[k]) for k in m.keys], dtype=float)
    x_ref, ref_info, ref_warn = _build_reference(m, reference, dataset_id)

    y_hat = m.predict_one(x)
    y_hat_ref = m.predict_one(x_ref)

    target = float(target_size)
    total_dev = y_hat - target
    param_attr = y_hat - y_hat_ref
    base_offset = y_hat_ref - target
    model_err = (float(observed_size) - y_hat) if observed_size is not None else None

    groups, group_info = _collinear_groups(m, dataset_id)

    warnings = []
    if ref_warn:
        warnings.append({"code": "REFERENCE_FALLBACK", "msg": ref_warn})
    if abs(base_offset) > 1.0:
        warnings.append({
            "code": "BASELINE_OFFSET_EXCEEDS_TOL",
            "value": round(float(base_offset), 4),
            "msg": f"参照工况本身预测值比标准低/高 {base_offset:+.2f}μm，"
                   f"建议先做模型偏差校准，否则参数归因占比会被稀释",
        })

    # ---- 贡献计算 ----
    ig = _ig(m, x, x_ref, ig_steps) if "ig" in methods else None
    oat = _oat(m, x, x_ref) if "oat" in methods else None
    phi = _group_shapley(m, x, x_ref, groups) if "group_shapley" in methods else None

    # 主口径：IG 优先，否则 OAT
    primary = ig if ig is not None else (oat if oat is not None else np.zeros(len(m.keys)))
    abs_primary = np.abs(primary)
    denom = abs_primary.sum()

    # 组级贡献（用于展示与共线说明）
    group_rows = []
    for gi, idxs in enumerate(groups):
        members = [m.keys[i] for i in idxs]
        g_ig = float(sum(primary[i] for i in idxs))
        g_phi = float(phi[gi]) if phi is not None else None
        group_rows.append({
            "group": "+".join(members) if len(members) > 1 else members[0],
            "members": members,
            "contribution_um": round(g_ig, 4),
            "shapley_um": round(g_phi, 4) if g_phi is not None else None,
            "collinear": len(members) > 1,
        })

    # 逐参数明细
    rows = []
    for i, k in enumerate(m.keys):
        rows.append({
            "name": k,
            "current": round(float(x[i]), 4),
            "reference": round(float(x_ref[i]), 4),
            "deviation": round(float(x[i] - x_ref[i]), 4),
            "contribution_um": round(float(primary[i]), 4),
            "share": round(float(abs_primary[i] / denom), 4) if denom > 0 else 0.0,
            "direction": "increase" if primary[i] > 0 else ("decrease" if primary[i] < 0 else "none"),
            "controllable": k in CONTROLLABLE_KEYS,
            "ig_um": round(float(ig[i]), 4) if ig is not None else None,
            "oat_um": round(float(oat[i]), 4) if oat is not None else None,
        })
    rows.sort(key=lambda r: -abs(r["contribution_um"]))
    rows = rows[:int(top_k)]

    # ---- 完备性校验 ----
    completeness = None
    if ig is not None:
        completeness = {
            "sum_ig": round(float(ig.sum()), 6),
            "parameter_attributable": round(float(param_attr), 6),
            "abs_error": float(abs(ig.sum() - param_attr)),
        }

    # ---- 方法一致性 ----
    agreement = None
    if ig is not None and oat is not None:
        order_ig = list(np.argsort(-np.abs(ig))[:3])
        order_oat = list(np.argsort(-np.abs(oat))[:3])
        agreement = {
            "ig_top3": [m.keys[i] for i in order_ig],
            "oat_top3": [m.keys[i] for i in order_oat],
            "same_top3": bool(set(order_ig) == set(order_oat)),
        }

    cf = _counterfactual(m, x, x_ref, target, top_k)
    if not cf:
        warnings.append({
            "code": "NO_FEASIBLE_SINGLE_PARAM",
            "msg": "单参数在训练区间内无法使预测达到标准值，需多参数协同（改用 /optsearch）",
        })

    result = {
        "model_id": model_id,
        "target_size": round(target, 4),
        "predicted_size": round(float(y_hat), 4),
        "observed_size": round(float(observed_size), 4) if observed_size is not None else None,
        "total_deviation": round(float(total_dev), 4),
        "decomposition": {
            "model_error": round(float(model_err), 4) if model_err is not None else None,
            "parameter_attributable": round(float(param_attr), 4),
            "baseline_offset": round(float(base_offset), 4),
        },
        "reference": {
            **ref_info,
            "predicted": round(float(y_hat_ref), 4),
            "values": {k: round(float(x_ref[i]), 4) for i, k in enumerate(m.keys)},
        },
        "contributions": rows,
        "group_contributions": sorted(group_rows, key=lambda r: -abs(r["contribution_um"])),
        "collinear_info": group_info,
        "counterfactual": cf,
        "methods_used": [x for x in methods if x in ("ig", "group_shapley", "oat")],
        "completeness": completeness,
        "method_agreement": agreement,
        "warnings": warnings,
    }

    log.info("归因完成 | model=%s target=%.2f 预测=%.4f | 总偏差=%+.4f "
             "可归因=%+.4f 基准偏移=%+.4f | Top1=%s",
             model_id, target, y_hat, total_dev, param_attr, base_offset,
             rows[0]["name"] if rows else "-")
    return result
