"""
统一方法对比结果文件。
所有方法（B1、B2、GAIL、CDQN 等）的指标都写入 method_comparison.csv，
便于在一张表里对比。
"""
import os
import pandas as pd

COMPARISON_FILE = "method_comparison.csv"
COLUMNS = ["method", "avg_timesteps", "avg_constraints_broken", "avg_waiting_time", 
           "min_waiting_time", "max_waiting_time"]


def _row_from_dict(method_name: str, d: dict) -> list:
    return [
        method_name,
        d.get("avg_timesteps", 0),
        d.get("avg_constraints_broken", 0),
        d.get("avg_waiting_time", 0),
        d.get("min_waiting_time", 0),
        d.get("max_waiting_time", 0),
    ]


def read_comparison() -> pd.DataFrame:
    """读取现有对比文件，若不存在则返回空 DataFrame。"""
    if not os.path.isfile(COMPARISON_FILE):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(COMPARISON_FILE)
    if list(df.columns) != COLUMNS:
        return pd.DataFrame(columns=COLUMNS)
    return df


def write_comparison(df: pd.DataFrame) -> None:
    """写入对比文件。"""
    df.to_csv(COMPARISON_FILE, index=False)
    print(f"结果已写入: {COMPARISON_FILE}")


def update_methods(method_results: dict) -> None:
    """
    更新或追加若干方法的指标，其余方法保留。
    method_results: {"B1": {...}, "B2": {...}} 或 {"GAIL": {...}} 等。
    """
    df = read_comparison()
    methods_to_update = set(method_results.keys())
    # 保留未在本次更新中的方法
    if len(df) > 0:
        df = df[~df["method"].isin(methods_to_update)]
    # 追加本次结果
    for name, metrics in method_results.items():
        row = _row_from_dict(name, metrics)
        df = pd.concat([df, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
    write_comparison(df)


def append_method(method_name: str, metrics: dict) -> None:
    """追加或覆盖单条方法结果（如 GAIL、CDQN）。"""
    update_methods({method_name: metrics})


# 与 simulation-results.csv 合并到一张表
SIMULATION_RESULTS_FILE = "simulation-results.csv"
COMPARISON_ALL_FILE = "comparison_all.csv"


def merge_simulation_with_baseline(
    baseline_path: str = COMPARISON_FILE,
    simulation_path: str = SIMULATION_RESULTS_FILE,
    out_path: str = COMPARISON_ALL_FILE,
) -> pd.DataFrame:
    """
    把 baseline（method_comparison.csv）和 simulation-results.csv 合并成一张表，
    写入 comparison_all.csv，便于在一张表里对比 B1、B2 与各 CDQN/DQN 配置。
    """
    baseline_path = baseline_path or COMPARISON_FILE
    simulation_path = simulation_path or SIMULATION_RESULTS_FILE
    out_path = out_path or COMPARISON_ALL_FILE

    # 读取 baseline 侧（B1、B2、GAIL 等）
    if os.path.isfile(baseline_path):
        df_base = pd.read_csv(baseline_path)
        if list(df_base.columns) != COLUMNS:
            df_base = pd.DataFrame(columns=COLUMNS)
    else:
        df_base = pd.DataFrame(columns=COLUMNS)

    # 读取 simulation 侧
    if not os.path.isfile(simulation_path):
        df_base.to_csv(out_path, index=False)
        print(f"未找到 {simulation_path}，仅写入 baseline 到 {out_path}")
        return df_base

    df_sim = pd.read_csv(simulation_path)
    rows = []
    for _, r in df_sim.iterrows():
        method = "{}_s{}c{}w{}mw{}t{}".format(
            str(r["model_type"]).upper(), r["s"], r["c"], r["w"], r["mw"], r["t"]
        )
        rows.append({
            "method": method,
            "avg_timesteps": r["avg_t_frame"],
            "avg_constraints_broken": r["avg_c_broken"],
            "avg_waiting_time": r["avg_waiting_time"],
            "min_waiting_time": r["min_waiting_time"],
            "max_waiting_time": r["max_waiting_time"],
        })
    df_sim_mapped = pd.DataFrame(rows, columns=COLUMNS)

    merged = pd.concat([df_base, df_sim_mapped], ignore_index=True)
    merged.to_csv(out_path, index=False)
    print(f"baseline + simulation 已合并到: {out_path}（共 {len(merged)} 行）")
    return merged
