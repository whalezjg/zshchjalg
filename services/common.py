"""
common.py —— 公共工具：路径、数据读取、参数 schema、列解析、喷嘴解析、评估指标

对外接口字段统一使用英文参数名（key），含义见 PARAM_SCHEMA 注释；
英文 key 与数据集中文列名之间的映射集中在 PARAM_SCHEMA，服务内部自动换算。
"""
import os
import re
import pickle
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_ROOT = os.path.join(BASE_DIR, "weight")

RANDOM_SEED = 34
TEST_SIZE = 0.2

# ============================================================
# 参数 Schema（对外英文 key  <=>  数据集中文列）
# ------------------------------------------------------------
# key       : 对外接口使用的英文参数名
# label     : 数据集中的标准中文列名（用于精确匹配）
# aliases   : 匹配数据集列时可用的关键词（命中任一即可）
# role      : model=模型特征 / display=前端展示(不参与模型) / target=目标列
# name_en   : 英文含义注释
# name_zh   : 中文含义注释
# unit      : 单位（展示用）
# ============================================================

DCS_PARAMS = [
    # 喷雾风机变频
    dict(key="sic3203", label="喷雾风机变频 SIC3203(Hz)", aliases=["SIC3203"],
         role="model", name_en="spray air blower frequency", name_zh="喷雾风机变频", unit="Hz"),
    # 细粉风机变频
    dict(key="sic1014", label="细粉风机变频SIC1014(Hz)", aliases=["SIC1014"],
         role="model", name_en="fine powder blower frequency", name_zh="细粉风机变频", unit="Hz"),
    # 回用风机变频
    dict(key="sic3204", label="回用风机变频SIC3204(Hz)", aliases=["SIC3204"],
         role="model", name_en="recycle blower frequency", name_zh="回用风机变频", unit="Hz"),
    # 喷雾压力
    dict(key="prc3101", label="喷雾压力PRC3101(MPa)", aliases=["PRC3101"],
         role="model", name_en="spray pressure", name_zh="喷雾压力", unit="MPa"),
    # 喷雾流量
    dict(key="fi3108", label="喷雾流量FI3108(m3/h)", aliases=["FI3108"],
         role="model", name_en="spray flow rate", name_zh="喷雾流量", unit="m3/h"),
    # 细粉分级器变频1
    dict(key="sic1008", label="细粉分级器变频SIC1008(Hz)", aliases=["SIC1008"],
         role="model", name_en="classifier frequency 1", name_zh="细粉分级器变频1", unit="Hz"),
    # 细粉分级器变频2
    dict(key="sic1009", label="细粉分级器变频SIC1009(Hz)", aliases=["SIC1009"],
         role="model", name_en="classifier frequency 2", name_zh="细粉分级器变频2", unit="Hz"),
]

QUALITY_PARAMS = [
    # 胶体固含量
    dict(key="solid_content", label="胶体固含量", aliases=["胶体固含量"],
         role="model", name_en="colloid solid content", name_zh="胶体固含量", unit=None),
    # 喷枪数量
    dict(key="gun_count", label="喷枪数量", aliases=["喷枪数量"],
         role="model", name_en="number of spray guns", name_zh="喷枪数量", unit="个"),
    # 黏度
    dict(key="viscosity", label="黏度", aliases=["黏度"],
         role="model", name_en="viscosity", name_zh="黏度", unit=None),
]

# 前端展示参数：不参与模型训练与预测，仅在请求/回显中出现
DISPLAY_PARAMS = [
    # 喷嘴 3.1 口径数量
    dict(key="nozzle_31_count", label="喷嘴3.1数量", aliases=["3.1", "3.1口径", "口径一"],
         role="display", name_en="nozzle count of caliber 3.1", name_zh="喷嘴3.1数量", unit="个"),
    # 喷嘴 3.3 口径数量
    dict(key="nozzle_33_count", label="喷嘴3.3数量", aliases=["3.3", "3.3口径", "口径二"],
         role="display", name_en="nozzle count of caliber 3.3", name_zh="喷嘴3.3数量", unit="个"),
]

# 粒度分布目标列（5 档体积分数 %）
PSD_TARGETS = [
    dict(key="psd_0_20", label="粒度分布(0μm～20μm)___%(体积分数)",
         aliases=["0μm～20μm", "0-20μm", "20μm"],
         role="target", name_en="PSD 0-20um volume fraction", name_zh="粒度分布(0-20μm)", unit="%"),
    dict(key="psd_0_40", label="粒度分布(0μm～40μm)___%(体积分数)",
         aliases=["0μm～40μm", "0-40μm", "40μm"],
         role="target", name_en="PSD 0-40um volume fraction", name_zh="粒度分布(0-40μm)", unit="%"),
    dict(key="psd_0_80", label="粒度分布(0μm～80μm)___%(体积分数)",
         aliases=["0μm～80μm", "0-80μm", "80μm"],
         role="target", name_en="PSD 0-80um volume fraction", name_zh="粒度分布(0-80μm)", unit="%"),
    dict(key="psd_0_105", label="粒度分布(0μm～105μm)___%(体积分数)",
         aliases=["0μm～105μm", "0-105μm", "105μm"],
         role="target", name_en="PSD 0-105um volume fraction", name_zh="粒度分布(0-105μm)", unit="%"),
    dict(key="psd_0_149", label="粒度分布(0μm～149μm)___%(体积分数)",
         aliases=["0μm～149μm", "0-149μm", "149μm"],
         role="target", name_en="PSD 0-149um volume fraction", name_zh="粒度分布(0-149μm)", unit="%"),
]

# 目标列参数（用于相关分析 / 文档说明）；含粒径、能耗与 5 档粒度分布
TARGET_PARAMS = [
    # 目标：粒径
    dict(key="particle_size", label="D(V,0.5)___μm", aliases=["D(V,0.5)"],
         role="target", name_en="particle size D(V,0.5)", name_zh="粒径D(V,0.5)", unit="μm"),
    # 目标：能源消耗（数据库 fcc_dataset_row 中列名为 energy）
    dict(key="energy_consumption", label="能源消耗", aliases=["能源消耗", "energy"],
         role="target", name_en="energy consumption", name_zh="能源消耗", unit=None),
] + PSD_TARGETS

# 模型特征（顺序即 SVR 输入顺序）：DCS 7 + 质量 3 = 10 个
MODEL_PARAMS = DCS_PARAMS + QUALITY_PARAMS
MODEL_KEYS = [p["key"] for p in MODEL_PARAMS]

# 全集
ALL_PARAMS = MODEL_PARAMS + DISPLAY_PARAMS + TARGET_PARAMS
PARAM_BY_KEY = {p["key"]: p for p in ALL_PARAMS}

# 粒度分布英文 key（顺序即预测输出顺序）
PSD_KEYS = [p["key"] for p in PSD_TARGETS]

# 目标 key -> (模型 pkl 文件名, scaler pkl 文件名)
# particle/energy 沿用历史命名以兼容已存在的旧模型
TARGET_MODEL_FILES = {
    "particle_size": ("particle_svr.pkl", "scaler_y.pkl"),
    "energy_consumption": ("energy_svr.pkl", "scaler_energy.pkl"),
    "psd_0_20": ("psd_0_20_svr.pkl", "scaler_psd_0_20.pkl"),
    "psd_0_40": ("psd_0_40_svr.pkl", "scaler_psd_0_40.pkl"),
    "psd_0_80": ("psd_0_80_svr.pkl", "scaler_psd_0_80.pkl"),
    "psd_0_105": ("psd_0_105_svr.pkl", "scaler_psd_0_105.pkl"),
    "psd_0_149": ("psd_0_149_svr.pkl", "scaler_psd_0_149.pkl"),
}

# SVR 网格搜索参数
SVR_PARAM_GRID = {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]}

# 离散（需取整）的特征英文 key：喷枪数量
INT_FEATURE_KEYS = {"gun_count"}


# ---- 目录与路径 ----
def model_dir(model_id: str) -> str:
    return os.path.join(WEIGHT_ROOT, model_id)


def gen_model_id() -> str:
    """14位时间戳模型id: 20260904093500"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M%S")


# ---- 数据库配置（默认值按前端提供的库写死，可用环境变量覆盖）----
# 宿主机直跑：127.0.0.1:3307；容器内运行：把 DB_HOST 指向宿主机网关或 MySQL 容器名
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3307")),
    "user": os.getenv("DB_USER", "zsh"),
    "password": os.getenv("DB_PASSWORD", "Zsh1234@"),
    "database": os.getenv("DB_NAME", "zsh_fcc"),
    "charset": "utf8mb4",
    "connect_timeout": 5,
}
DB_DATASET_TABLE = os.getenv("DB_DATASET_TABLE", "fcc_dataset_row")


def load_dataset_from_db(dataset_id):
    """
    按 dataset_id 从 MySQL 数据集明细表读取样本，返回 DataFrame（列名为英文 key）

    dataset_id = 0 表示不按数据集过滤，读取整张表的全部行。
    """
    import pymysql

    dataset_id = int(dataset_id)
    if dataset_id == 0:
        sql = f"SELECT * FROM `{DB_DATASET_TABLE}` ORDER BY sample_shift, id"
        params = None
    else:
        sql = (f"SELECT * FROM `{DB_DATASET_TABLE}` "
               f"WHERE dataset_id = %s ORDER BY sample_shift, id")
        params = (dataset_id,)

    try:
        conn = pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **DB_CONFIG)
    except Exception as e:
        raise RuntimeError(
            f"cannot connect to MySQL {DB_CONFIG['host']}:{DB_CONFIG['port']}: {e}"
        ) from e
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        if dataset_id == 0:
            raise ValueError(f"no rows in `{DB_DATASET_TABLE}`")
        raise ValueError(f"dataset_id={dataset_id} has no rows in `{DB_DATASET_TABLE}`")
    return pd.DataFrame(rows)


def latest_dataset_id():
    """返回 fcc_dataset_row 中最大的 dataset_id；表中无数据时抛错"""
    import pymysql

    try:
        conn = pymysql.connect(**DB_CONFIG)
    except Exception as e:
        raise RuntimeError(
            f"cannot connect to MySQL {DB_CONFIG['host']}:{DB_CONFIG['port']}: {e}"
        ) from e
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(dataset_id) FROM `{DB_DATASET_TABLE}`")
            row = cur.fetchone()
    finally:
        conn.close()

    if not row or row[0] is None:
        raise ValueError(f"no dataset found in `{DB_DATASET_TABLE}`")
    return int(row[0])


def is_dataset_ref(source):
    """判断数据来源是否为数据库数据集: int、纯数字字符串、或 'dataset:1' 形式"""
    if isinstance(source, bool):
        return False
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        s = source.strip()
        return s.startswith("dataset:") or s.isdigit()
    return False


def dataset_id_of(source):
    """从数据集引用中取出 dataset_id"""
    if isinstance(source, int):
        return source
    return int(str(source).strip().split(":", 1)[-1])


# ---- 数据读取（支持 csv / xlsx / 数据库数据集）----
def read_dataframe(path):
    """
    数据来源支持两种:
        - dataset_id: int / 纯数字字符串 / 'dataset:1' → 查 MySQL fcc_dataset_row
          其中 dataset_id=0 表示读取整张表的全部行（不按数据集过滤）
        - 文件路径: .xlsx / .csv（兼容原有本地文件方式）
    """
    if is_dataset_ref(path):
        return load_dataset_from_db(dataset_id_of(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"data file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        return pd.read_excel(path)
    for enc in ["utf-8-sig", "gbk", "gb2312", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"cannot decode file: {path}")


# ---- 名称匹配工具 ----
def _norm(s):
    """归一化：去空白、全半角括号/引号、下划线、中划线，统一小写"""
    if s is None:
        return ""
    s = str(s)
    for ch in ["（", "）", "(", ")", " ", "　", "_", "—", "-", '"', "'", "“", "”", "，", ",", "：", ":", "；", ";"]:
        s = s.replace(ch, "")
    return s.lower()


def _key_of_col(col):
    """由数据集列名反向查找英文 key，找不到返回 None（英文列名直接命中）"""
    nc = _norm(col)
    for p in ALL_PARAMS:
        if _norm(p["key"]) == nc:
            return p["key"]
    for p in MODEL_PARAMS + TARGET_PARAMS:
        if _norm(p["label"]) == nc:
            return p["key"]
    for p in ALL_PARAMS:
        for a in p["aliases"]:
            if _norm(a) and _norm(a) in nc:
                return p["key"]
    return None


def find_column(df, param_schema_item, required=True):
    """
    按 schema 条目在 DataFrame 中匹配实际列名。
    匹配优先级: 英文 key 精确 > 中文 label 精确 > alias 包含（取最短）。
    返回数据集真实列名。
    """
    # 数据库表 fcc_dataset_row 直接以英文 key 作列名，此处优先精确命中
    key_norm = _norm(param_schema_item["key"])
    key_hits = [c for c in df.columns if _norm(c) == key_norm]
    if key_hits:
        return key_hits[0]

    nl = _norm(param_schema_item["label"])
    exact = [c for c in df.columns if _norm(c) == nl]
    hits = exact if exact else [
        c for c in df.columns
        if any(_norm(a) and _norm(a) in _norm(c) for a in param_schema_item["aliases"])
    ]
    if hits:
        return sorted(hits, key=lambda c: len(str(c)))[0]
    if required:
        raise ValueError(
            f"column not found for key '{param_schema_item['key']}' "
            f"(label: {param_schema_item['label']}). available: {list(df.columns)}"
        )
    return None


def resolve_model_feature_columns(df):
    """按 MODEL_PARAMS 顺序解析 10 个模型特征列，返回数据集实际列名"""
    cols = []
    for item in MODEL_PARAMS:
        cols.append(find_column(df, item, required=True))
    return cols


def resolve_target_columns(df):
    """解析目标列：particle_size / energy_consumption，返回 {英文key: 实际列名}"""
    targets = {}
    for item in TARGET_PARAMS:
        targets[item["key"]] = find_column(df, item, required=True)
    return targets


def col_english_key(col):
    """数据集中文列名 -> 对外英文 key"""
    key = _key_of_col(col)
    return key if key is not None else col


# ---- 喷嘴解析（用于相关性分析时把文本列数值化）----
def parse_nozzle_text(text):
    """把 '3.3 8个' / '3.1 5个 3.3 3个' 解析成 {口径: 数量}"""
    if pd.isna(text):
        return {}
    text = str(text)
    out = {}
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(\d+)\s*[个副支套把]", text):
        dia = m.group(1)
        cnt = int(m.group(2))
        out.setdefault(dia, 0)
        out[dia] += cnt
    return out


def is_numeric_series(s):
    return pd.to_numeric(s, errors="coerce").notna().sum() > 0


# ---- 评估指标 ----
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
