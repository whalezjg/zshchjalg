"""
optimizer.py —— 寻优核心（加权粒子群 / 多目标粒子群）

参考 D:/project/zshchj/fccchjalg 的 PSO 设计，基于指定 model_id 的
粒径/能耗 SVR 与 scaler 进行工艺参数寻优。

寻优目标:
    obj1 = |预测粒径 - 目标粒径|   （尽量贴近 target_size）
    obj2 = 预测能耗                （尽量低）

算法:
    weighted —— 加权粒子群: fitness = α*obj1_norm + (1-α)*obj2_norm
    mopso    —— 多目标粒子群: Pareto 档案 + 拥挤度剪枝，返回前沿解集

对外 params 配置使用英文 key（见 common.PARAM_SCHEMA），
例如 sic3203 / prc3101 / solid_content / gun_count / viscosity。
锁定参数(locked=true)固定为 value；未锁定参数在 [lower, upper] 内寻优。
喷嘴等展示参数(nozzle_31_count/nozzle_33_count)不参与模型，仅回显。

对外字段（返回）均为英文：
    model_id / algorithm / alpha / target_size
    optimized_parameters: [{name, value, locked}]
    predicted_particle_size_um / predicted_energy_consumption / pareto_solutions
"""
import numpy as np
from services.predictor import RunBundle
from services.common import PARAM_BY_KEY, MODEL_KEYS
from services.log_util import get_logger

log = get_logger("optimizer")

W, C1, C2 = 0.5, 2.0, 2.0       # PSO 超参（惯性/个体/全局学习因子）
DEFAULT_ARCHIVE = 60            # Pareto 档案容量
EPS = 1e-9
INT_FEATURE_KEYS = {"gun_count"}  # 离散特征（喷枪数量取整）


def _round_int_feature(x, keys):
    """离散特征取整（gun_count -> 整数），支持 1D / 2D"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        for i, k in enumerate(keys):
            if k in INT_FEATURE_KEYS:
                x[i] = np.round(x[i])
        return x
    for i, k in enumerate(keys):
        if k in INT_FEATURE_KEYS:
            x[:, i] = np.round(x[:, i])
    return x


class ParamSolver:
    """按参数配置在模型特征空间求解"""

    def __init__(self, bundle: RunBundle, param_configs: list):
        self.bundle = bundle
        # 特征英文 key 顺序（与特征列一一对应）
        self.feature_keys = bundle.config.get("feature_key_order") or list(MODEL_KEYS)
        self.n_features = len(self.feature_keys)

        self.lb = np.full(self.n_features, np.nan)
        self.ub = np.full(self.n_features, np.nan)
        self.fixed = np.zeros(self.n_features, dtype=bool)
        self.fixed_val = np.zeros(self.n_features)
        self.param_map = {}       # feature key -> config
        self.display = []         # 不参与寻优的展示参数（喷嘴等）

        bounds_default = bundle.feature_bounds  # {数据集列名: [min,max]}
        col_by_key = dict(zip(self.feature_keys, bundle.feature_columns))

        # 先解析全部配置
        for cfg in param_configs:
            name = cfg.get("name", "")
            if name in self.feature_keys:
                self.param_map[name] = cfg
            else:
                # 非模型特征（如喷嘴展示参数 nozzle_31_count 等）仅回显
                self.display.append(cfg)

        # 逐特征生成边界/固定值
        for i, key in enumerate(self.feature_keys):
            col = col_by_key[key]
            lo_d, hi_d = bounds_default.get(col, [None, None])
            cfg = self.param_map.get(key)
            if cfg is None:
                if lo_d is None or hi_d is None:
                    raise ValueError(f"feature '{key}'({col}) has no config and no data bounds")
                self.lb[i], self.ub[i] = float(lo_d), float(hi_d)
                continue

            locked = bool(cfg.get("locked", False))
            has_val = cfg.get("value") is not None
            has_bounds = cfg.get("lower") is not None or cfg.get("upper") is not None
            if locked or (has_val and not has_bounds):
                v = float(cfg["value"])
                self.fixed[i] = True
                self.fixed_val[i] = v
                self.lb[i] = self.ub[i] = v
            else:
                lo = cfg.get("lower", lo_d)
                hi = cfg.get("upper", hi_d)
                if lo is None or hi is None:
                    raise ValueError(f"parameter '{key}' unlocked but no lower/upper")
                self.lb[i], self.ub[i] = float(lo), float(hi)

        if np.any(np.isnan(self.lb)) or np.any(np.isnan(self.ub)):
            raise ValueError("some features have no valid bounds")

    def _snap(self, X):
        """限制边界 + 锁定固定 + 离散取整（1D/2D 兼容）"""
        X = np.clip(X, self.lb, self.ub)
        if self.fixed.any():
            X = X.copy()
            if X.ndim == 1:
                X[self.fixed] = self.fixed_val[self.fixed]
            else:
                X[:, self.fixed] = self.fixed_val[self.fixed]
        X = _round_int_feature(X, self.feature_keys)
        return np.clip(X, self.lb, self.ub)

    def _predict_batch(self, X_raw):
        """批量预测（返回真实尺度粒径/能耗数组）"""
        Xs = self.bundle.scaler_X.transform(X_raw)
        ps = self.bundle.particle_model.predict(Xs)
        es = self.bundle.energy_model.predict(Xs)
        size = self.bundle.scaler_y.inverse_transform(ps.reshape(-1, 1)).ravel()
        energy = self.bundle.scaler_energy.inverse_transform(es.reshape(-1, 1)).ravel()
        return size, energy

    # ---------- 内部优化器 ----------
    def _optimize(self, algorithm, alpha, target_size, max_iter, num_particles):
        D = self.n_features
        X = self._snap(np.random.uniform(self.lb, self.ub, size=(num_particles, D)))
        V = np.zeros_like(X)
        size0, energy0 = self._predict_batch(X)
        err0 = np.abs(size0 - target_size)

        if algorithm == "weighted":
            e_min, e_max = energy0.min(), energy0.max()
            err_min, err_max = err0.min(), err0.max()
            en_span, er_span = e_max - e_min + EPS, err_max - err_min + EPS

            def fitness(err, energy):
                fn = (energy - e_min) / en_span
                sn = (err - err_min) / er_span
                return alpha * sn + (1 - alpha) * fn

            pbest_X = X.copy()
            pbest_f = fitness(err0, energy0)
            g_idx = int(np.argmin(pbest_f))
            gbest_X = X[g_idx].copy()
            gbest_f = pbest_f[g_idx]

            for _ in range(max_iter):
                r1, r2 = np.random.rand(num_particles, D), np.random.rand(num_particles, D)
                V = W * V + C1 * r1 * (pbest_X - X) + C2 * r2 * (gbest_X - X)
                X = self._snap(X + V)
                sz, en = self._predict_batch(X)
                er = np.abs(sz - target_size)
                f = fitness(er, en)
                better = f < pbest_f
                pbest_X[better] = X[better]
                pbest_f[better] = f[better]
                gi = int(np.argmin(f))
                if f[gi] < gbest_f:
                    gbest_X, gbest_f = X[gi].copy(), f[gi]

            best_raw = gbest_X
            best_size, best_energy = self._predict_batch(best_raw.reshape(1, -1))
            return best_raw, float(best_size[0]), float(best_energy[0]), None

        # ---- MOPSO ----
        archive = []
        pbest_X = X.copy()
        pbest_obj = np.stack([err0, energy0], axis=1)

        def dominates(a, b):
            return np.all(a <= b - EPS) or (np.all(a <= b + EPS) and np.any(a < b - EPS))

        def update_archive(new_obj, new_pos):
            nonlocal archive
            to_remove = []
            for i, (o, p) in enumerate(archive):
                if dominates(new_obj, o):
                    to_remove.append(i)
                elif dominates(o, new_obj):
                    return
            archive = [it for idx, it in enumerate(archive) if idx not in to_remove]
            archive.append((new_obj.copy(), new_pos.copy()))
            if len(archive) > DEFAULT_ARCHIVE:
                archive = _crowding_prune(archive, DEFAULT_ARCHIVE)

        for i in range(num_particles):
            update_archive(pbest_obj[i], pbest_X[i])

        for _ in range(max_iter):
            for i in range(num_particles):
                if archive:
                    g = archive[int(np.random.randint(len(archive)))][1]
                else:
                    g = pbest_X[i]
                r1, r2 = np.random.rand(D), np.random.rand(D)
                V[i] = W * V[i] + C1 * r1 * (pbest_X[i] - X[i]) + C2 * r2 * (g - X[i])
                X[i] = self._snap(X[i] + V[i])

            sz, en = self._predict_batch(X)
            objs = np.stack([np.abs(sz - target_size), en], axis=1)
            for i in range(num_particles):
                if dominates(objs[i], pbest_obj[i]) or np.allclose(objs[i], pbest_obj[i]):
                    pbest_obj[i] = objs[i].copy()
                    pbest_X[i] = X[i].copy()
                update_archive(objs[i], X[i])

        archive = _crowding_prune(archive, DEFAULT_ARCHIVE * 2)
        archive.sort(key=lambda t: (t[0][1], t[0][0]))
        front = []
        for o, p in archive:
            sz, en = self._predict_batch(p.reshape(1, -1))
            front.append({
                "optimized_parameters": p.tolist(),
                "particle_size_um": round(float(sz[0]), 4),
                "energy_consumption": round(float(en[0]), 4),
                "size_error": round(float(o[0]), 4),
            })
        if archive:
            _, best = min(
                archive,
                key=lambda t: alpha * (t[0][0] / (t[0][0] + t[0][1] + EPS))
                + (1 - alpha) * (t[0][1] / (t[0][0] + t[0][1] + EPS)),
            )
            best_raw = best
            best_size, best_energy = self._predict_batch(best_raw.reshape(1, -1))
        else:
            best_raw = pbest_X[0]
            best_size, best_energy = self._predict_batch(best_raw.reshape(1, -1))
        return best_raw, float(best_size[0]), float(best_energy[0]), front

    # ---------- 对外接口 ----------
    def solve(self, algorithm, alpha, target_size, max_iter=200, num_particles=100):
        if algorithm not in ("weighted", "mopso"):
            raise ValueError(f"unknown algorithm: {algorithm} (weighted / mopso)")
        alpha = float(alpha)
        if not (0 <= alpha <= 1):
            raise ValueError("alpha (particle weight) must be in [0,1]")

        best_raw, best_size, best_energy, pareto = self._optimize(
            algorithm, alpha, target_size, int(max_iter), int(num_particles),
        )
        best_raw = best_raw.tolist()
        log.info("寻优完成 | algorithm=%s alpha=%.2f target_size=%s | 预测粒径=%.4f 预测能耗=%.4f | 前沿解=%s",
                 algorithm, alpha, target_size, best_size, best_energy,
                 len(pareto) if pareto else 0)

        optimized = []
        for key, v in zip(self.feature_keys, best_raw):
            cfg = self.param_map.get(key)
            optimized.append({
                "name": key,
                "value": round(float(v), 4),
                "locked": bool(cfg.get("locked", False)) if cfg else False,
            })
        for cfg in self.display:  # 展示参数原样回显（英文 key）
            optimized.append({
                "name": cfg.get("name", ""),
                "value": cfg.get("value", None),
                "locked": bool(cfg.get("locked", False)),
            })

        return {
            "algorithm": algorithm,
            "alpha": alpha,
            "target_size": float(target_size),
            "optimized_parameters": optimized,
            "predicted_particle_size_um": round(best_size, 4),
            "predicted_energy_consumption": round(best_energy, 4),
            "pareto_solutions": pareto,
        }


def _crowding_prune(archive, max_size):
    """按拥挤度剪枝 Pareto 档案"""
    if len(archive) <= max_size:
        return archive
    n = len(archive)
    n_obj = 2
    dist = np.zeros(n)
    for dim in range(n_obj):
        order = np.argsort([it[0][dim] for it in archive])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        rng = archive[order[-1]][0][dim] - archive[order[0]][0][dim] + EPS
        for k in range(1, n - 1):
            dist[order[k]] += (archive[order[k + 1]][0][dim] - archive[order[k - 1]][0][dim]) / rng
    keep = np.argsort(dist)[::-1][:max_size]
    return [archive[i] for i in sorted(keep)]


def optimize(model_id, algorithm, alpha, target_size, max_iter=200,
             num_particles=100, param_configs=None):
    """执行寻优，返回结构见 ParamSolver.solve()"""
    log.info("寻优开始 | model_id=%s algorithm=%s alpha=%s target_size=%s max_iter=%s num_particles=%s | 参数配置=%s",
             model_id, algorithm, alpha, target_size, max_iter, num_particles, param_configs)
    bundle = RunBundle(model_id)
    solver = ParamSolver(bundle, param_configs or [])
    log.info("特征边界已生成 | 特征=%s lb=%s ub=%s", solver.feature_keys,
             solver.lb.tolist(), solver.ub.tolist())
    result = solver.solve(algorithm, alpha, target_size, max_iter, num_particles)
    result["model_id"] = model_id
    return result
