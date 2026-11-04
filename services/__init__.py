"""
services —— HTTP 服务核心模块

目录结构:
    weight/<model_id>/            # 单次训练产物（模型+scaler+配置）
        config.json               # 训练元信息与测试指标
        feature_columns.json      # 模型特征列（顺序即 SVR 输入顺序）
        feature_bounds.json       # 各特征列原始 min/max（供寻优兜底边界）
        particle_svr.pkl          # 粒径 SVR 模型
        energy_svr.pkl            # 能耗 SVR 模型
        scaler_X.pkl              # 特征标准化
        scaler_y.pkl              # 粒径目标标准化
        scaler_energy.pkl         # 能耗目标标准化

模型特征（10 个，DCS 7 + 质量 3）:
    喷雾风机变频 SIC3203(Hz) / 细粉风机变频SIC1014(Hz) / 回用风机变频SIC3204(Hz)
    喷雾压力PRC3101(MPa) / 喷雾流量FI3108(m3/h)
    细粉分级器变频SIC1008(Hz) / 细粉分级器变频SIC1009(Hz)
    胶体固含量 / 喷枪数量 / 黏度
注: "喷嘴3.1数量 / 喷嘴3.3数量" 仅为前端展示参数，不参与训练与预测。
"""
