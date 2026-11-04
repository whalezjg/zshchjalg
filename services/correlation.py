"""
correlation.py —— 相关性分析核心

入参: data_path + params（英文参数 key 列表，个数不限）
出参: columns(英文 key) + matrix + notes

params 支持的英文 key（见 common.PARAM_SCHEMA）:
    - 模型特征: sic3203 / sic1014 / sic3204 / prc3101 / fi3108 /
                sic1008 / sic1009 / solid_content / gun_count / viscosity
    - 喷嘴口径数量: nozzle_31_count / nozzle_33_count（数据集缺失时自动解析"喷嘴搭配"文本）
    - 目标列: particle_size(D(V,0.5)) / energy_consumption(能源消耗)
"""
import re
import numpy as np
import pandas as pd

from services.common import (
    read_dataframe, _norm, find_column, is_numeric_series,
    parse_nozzle_text, PARAM_BY_KEY,
)
from services.log_util import get_logger

log = get_logger("correlation")

# 展示/目标的 schema 条目在 find_column 时用别名即可；口径英文 key 生成规则
DIA_TO_KEY = {"3.0": "nozzle_30_count", "3.1": "nozzle_31_count",
              "3.3": "nozzle_33_count"}


def _dia_key(dia):
    return DIA_TO_KEY.get(dia, f"nozzle_{dia.replace('.', '')}_count")


def _find_nozzle_text_col(df):
    """返回数据集中的喷嘴搭配文本列（nozzle_combo 或列名含 喷嘴/口径，且非数值），无则 None"""
    for c in df.columns:
        if _norm(c) == _norm("nozzle_combo") and not is_numeric_series(df[c]):
            return c
        if ("喷嘴" in str(c) or "口径" in str(c)) and not is_numeric_series(df[c]):
            return c
    return None


def _expand_nozzle_columns(df, nozzle_col, only_dia=None):
    """
    把喷嘴搭配文本解析成 {口径:数量}，添加数值列到 df。
    返回 [(英文key, 数据集列名)]（附加到 df 的新列，列名为 {dia}口径个数）
    """
    parsed = df[nozzle_col].apply(parse_nozzle_text)
    all_dia = sorted({d for cnt_map in parsed for d in cnt_map})
    added = []
    for d in all_dia:
        if only_dia is not None and d != only_dia:
            continue
        col_name = f"{d}口径个数"
        df[col_name] = [cnt_map.get(d, 0) for cnt_map in parsed]
        added.append((_dia_key(d), col_name))
    return added


def resolve_param_columns(df, param_key):
    """
    把一个英文 key 解析为 1~N 个数值列。

    返回: (out_keys, col_names, note)
        out_keys  输出时使用的英文 key 列表（即 columns 返回值）
        col_names 实际加入 df 的数值列名（数据集中文列名，仅供内部拼接矩阵）
        note      英文解析状态说明（出参不包含中文）
    """
    item = PARAM_BY_KEY.get(param_key)
    if item is None:
        raise ValueError(
            f"unknown parameter key: '{param_key}'. "
            f"allowed keys: {sorted(PARAM_BY_KEY)}"
        )

    # ---- 喷嘴口径数量：优先已有数值列，否则解析文本 ----
    if item["role"] == "display":
        dia = None
        for a in item["aliases"]:
            m = re.search(r"(3\.\d)", a)
            if m:
                dia = m.group(1)
                break
        if dia:
            # 数据集已有该口径的数值列（列名含 3.1/3.3 的数字）
            for c in df.columns:
                if dia in str(c) and is_numeric_series(df[c]):
                    return [param_key], [c], f"ok ({param_key})"
        # 尝试解析喷嘴搭配文本列
        nozzle_col = _find_nozzle_text_col(df)
        if nozzle_col is not None:
            added = _expand_nozzle_columns(df, nozzle_col, only_dia=dia)
            if added:
                keys = [k for k, _ in added]
                cols = [c for _, c in added]
                return keys, cols, f"ok ({param_key}, parsed from nozzle text)"
        raise ValueError(f"cannot resolve nozzle parameter '{param_key}' in dataset")

    # ---- 模型特征 / 目标列：按 schema 匹配数值列 ----
    col = find_column(df, item, required=False)
    if col is None:
        raise ValueError(
            f"cannot find column for '{param_key}' (label: {item['label']})"
        )
    if not is_numeric_series(df[col]):
        raise ValueError(f"column matched by '{param_key}' is not numeric")
    return [param_key], [col], f"ok ({param_key})"


def correlation_analysis(data_path, params):
    """
    计算所选参数的相关性矩阵

    返回:
        columns(英文key), matrix, notes(英文)
    """
    df = read_dataframe(data_path)
    log.info("相关性分析开始 | data_path=%s | 行数=%d | 参数=%s", data_path, len(df), params)
    out_keys, col_names, notes = [], [], []
    for p in params:
        keys, cols, note = resolve_param_columns(df, p)
        out_keys.extend(keys)
        col_names.extend(cols)
        notes.append(f"[{p}] {note}")
    # 去重（保留顺序）
    seen = set()
    uniq_keys, uniq_cols = [], []
    for k, c in zip(out_keys, col_names):
        if c not in seen:
            seen.add(c)
            uniq_keys.append(k)
            uniq_cols.append(c)
    if len(uniq_cols) < 2:
        raise ValueError("need at least 2 numeric parameters for correlation analysis")

    sub = df[uniq_cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(how="any")
    if len(sub) < 3:
        raise ValueError(f"too few valid rows ({len(sub)}) for correlation matrix")

    corr = sub.corr()
    matrix = np.array(corr.fillna(0.0), dtype=float)
    np.fill_diagonal(matrix, 1.0)
    matrix = np.round(matrix, 4).tolist()

    log.info("相关性分析完成 | 列=%s | 有效行=%d | 矩阵=%s", uniq_keys, len(sub), matrix)
    return {
        "columns": uniq_keys,
        "matrix": matrix,
        "notes": notes,
    }
