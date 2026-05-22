#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward statistics script (Python 3.8+)

功能概述：
1) 解析测试结果 .txt 文件中的 reward 数值（默认取每行最后一个可解析的数字作为 reward）
2) 对每个文件计算 reward 的均值与方差（使用总体方差 pvariance）
3) 遍历 d:\\vscode\\src_rtc\\atla_test_results 目录下的直接子文件夹，汇总每个文件的统计结果
4) 将结构化结果同时打印到控制台，并保存为脚本同目录下的 reward_statistics_report.txt（保留 4 位小数）

使用方式（可选）：
python reward_statistics.py --root "d:\\vscode\\src_rtc\\atla_test_results"
若不传入 --root，则默认使用上述路径。
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
import statistics
# 优先使用 numpy 进行高效计算；若不可用则回退到 statistics
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False

class TeeLogger:
    """
    将 stdout / stderr 同时输出到终端和日志文件
    """
    def __init__(self, logfile_path: Path, stream):
        self.stream = stream          # 原始 stdout 或 stderr
        self.logfile = logfile_path.open("a", encoding="utf-8")

    def write(self, message):
        self.stream.write(message)
        self.logfile.write(message)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def close(self):
        try:
            self.logfile.close()
        except Exception:
            pass
        
def _is_float_token(token: str) -> bool:
    """判断一个字符串是否可解析为浮点数。"""
    try:
        float(token)
        return True
    except Exception:
        return False


def parse_reward_values(file_path: Path) -> List[float]:
    """
    从日志文件中解析 reward 数值列表。
    解析策略：
    - 跳过空行
    - 每行按空白分隔，优先从右向左查找第一个可解析为浮点的 token，作为 reward
     （适配 'time\tbitrate\tdelay\tloss\tsmooth\t...\treward' 的格式）
    - 若整行无法解析出任何浮点数，则跳过该行
    """
    rewards: List[float] = []
    # 使用 utf-8 打开；若失败再尝试 gbk 以增强兼容性
    encodings = ("utf-8", "gbk")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with file_path.open("r", encoding=enc) as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    for idx in range(len(tokens) - 1, -1, -1):
                        if _is_float_token(tokens[idx]):
                            try:
                                rewards.append(float(tokens[idx]))
                            except Exception:
                                # 极端情况下 float() 通过但转换异常，忽略该行
                                pass
                            break
                return rewards
        except Exception as e:
            last_err = e
            # 尝试下一种编码
            continue
    # 若所有编码都失败，则抛出最后一次异常
    if last_err:
        raise last_err
    return rewards


def parse_columns(file_path: Path) -> dict:
    """
    Parse columns 2-5 (bitrate, delay, packet_loss, smooth) from log file.
    Returns dict with keys: bitrate, delay, packet_loss, smooth.
    """
    cols = {'bitrate': [], 'delay': [], 'packet_loss': [], 'smooth': []}
    col_indices = {'bitrate': 1, 'delay': 2, 'packet_loss': 3, 'smooth': 4}
    encodings = ("utf-8", "gbk")
    last_err = None
    for enc in encodings:
        try:
            with file_path.open("r", encoding=enc) as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    if len(tokens) < 5:
                        continue
                    for col_name, col_idx in col_indices.items():
                        try:
                            cols[col_name].append(float(tokens[col_idx]))
                        except (ValueError, IndexError):
                            pass
            return cols
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return cols


def compute_mean_variance(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    计算均值与总体方差（pvariance）。若列表为空，返回 (None, None)。
    - 均值：statistics.mean
    - 方差：statistics.pvariance（总体方差，分母为 N）
    - 若可用，优先使用 numpy：mean = np.mean(x)，variance = np.var(x, ddof=0)
    """
    if not values:
        return None, None
    try:
        if _HAS_NUMPY:
            arr = np.asarray(values, dtype=float)
            mean_val = float(np.mean(arr))
            # 总体方差（Population Variance）：除以 N（ddof=0）
            var_val = float(np.var(arr, ddof=0))
        else:
            mean_val = statistics.mean(values)
            var_val = statistics.pvariance(values)
        return mean_val, var_val
    except Exception:
        return None, None


def format_float_4(value: Optional[float]) -> str:
    """将浮点数格式化为 4 位小数。None 返回 'N/A'。"""
    if value is None:
        return "N/A"
    try:
        return f"{value:.4f}"
    except Exception:
        return "N/A"


def find_direct_subfolders(root: Path) -> List[Path]:
    """查找 root 下的直接子文件夹（不递归）。"""
    subs: List[Path] = []
    try:
        for p in root.iterdir():
            if p.is_dir():
                subs.append(p)
    except Exception:
        pass
    return subs


def find_txt_files(folder: Path) -> List[Path]:
    """查找 folder 下的所有 .txt 文件（不递归）。"""
    files: List[Path] = []
    try:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() == ".txt":
                files.append(p)
    except Exception:
        pass
    return files


def build_table_rows(root: Path) -> List[str]:
    """
    遍历 root 的直接子文件夹，为每个 .txt 文件生成一行统计结果。
    行结构：folder_name | file_name | reward_mean | reward_variance
    同时为每个子文件夹（模型）生成“所有 trace 的总体均值与总体方差”汇总行，
    汇总行 File 列置为 '(ALL_TRACES)' 以与现有四列表头格式一致。
    """
    rows: List[str] = []
    subfolders = find_direct_subfolders(root)
    if not subfolders:
        rows.append(f"[INFO] 目录无直接子文件夹: {str(root)}")
        return rows

    header = (
        "Folder\tFile\t"
        "RewardMean\tRewardVar\t"
        "Worst10%Mean\tWorst10%Var\t"
        "BitrateMean\tBitrateVar\t"
        "DelayMean\tDelayVar\t"
        "PktLossMean\tPktLossVar\t"
        "SmoothMean\tSmoothVar"
    )
    rows.append(header)
    for sub in sorted(subfolders, key=lambda p: p.name):
        txt_files = find_txt_files(sub)
        # 累积当前模型的所有 reward，用于模型级别总体统计
        all_rewards_for_folder: List[float] = []
        all_cols_for_folder = {'bitrate': [], 'delay': [], 'packet_loss': [], 'smooth': []}
        if not txt_files:
            rows.append(f"{sub.name}\t(no_txt_files)" + "\tN/A" * 12)
            rows.append(f"{sub.name}\t(ALL_TRACES)" + "\tN/A" * 12)
            continue

        for txt in sorted(txt_files, key=lambda p: p.name):
            try:
                rewards = parse_reward_values(txt)
                col_data = parse_columns(txt)
            except FileNotFoundError:
                rows.append(f"{sub.name}\t{txt.name}" + "\t[ERROR]" * 12)
                continue
            except Exception as e:
                rows.append(f"{sub.name}\t{txt.name}" + f"\t[{type(e).__name__}]" * 12)
                continue

            mean_val, var_val = compute_mean_variance(rewards)
            w_mean, w_var = compute_worst_k_mean_variance(rewards, ratio=0.1)
            br_mean, br_var = compute_mean_variance(col_data['bitrate'])
            dl_mean, dl_var = compute_mean_variance(col_data['delay'])
            pl_mean, pl_var = compute_mean_variance(col_data['packet_loss'])
            sm_mean, sm_var = compute_mean_variance(col_data['smooth'])

            rows.append(
                f"{sub.name}\t{txt.name}\t"
                f"{format_float_4(mean_val)}\t{format_float_4(var_val)}\t"
                f"{format_float_4(w_mean)}\t{format_float_4(w_var)}\t"
                f"{format_float_4(br_mean)}\t{format_float_4(br_var)}\t"
                f"{format_float_4(dl_mean)}\t{format_float_4(dl_var)}\t"
                f"{format_float_4(pl_mean)}\t{format_float_4(pl_var)}\t"
                f"{format_float_4(sm_mean)}\t{format_float_4(sm_var)}"
            )
            # 累积到模型级汇总
            all_rewards_for_folder.extend(rewards)
            for k in all_cols_for_folder:
                all_cols_for_folder[k].extend(col_data[k])

        agg_mean, agg_var = compute_mean_variance(all_rewards_for_folder)
        agg_br_mean, agg_br_var = compute_mean_variance(all_cols_for_folder['bitrate'])
        agg_dl_mean, agg_dl_var = compute_mean_variance(all_cols_for_folder['delay'])
        agg_pl_mean, agg_pl_var = compute_mean_variance(all_cols_for_folder['packet_loss'])
        agg_sm_mean, agg_sm_var = compute_mean_variance(all_cols_for_folder['smooth'])
        rows.append(
            f"{sub.name}\t(ALL_TRACES)\t"
            f"{format_float_4(agg_mean)}\t{format_float_4(agg_var)}\t"
            f"N/A\tN/A\t"
            f"{format_float_4(agg_br_mean)}\t{format_float_4(agg_br_var)}\t"
            f"{format_float_4(agg_dl_mean)}\t{format_float_4(agg_dl_var)}\t"
            f"{format_float_4(agg_pl_mean)}\t{format_float_4(agg_pl_var)}\t"
            f"{format_float_4(agg_sm_mean)}\t{format_float_4(agg_sm_var)}"
        )
    return rows


def build_aggregate_rows(root: Path) -> List[str]:
    rows: List[str] = []
    header = "Dataset\tFolder\tRewardMean\tRewardVariance"
    rows.append(header)
    subfolders = find_direct_subfolders(root)
    if not subfolders:
        rows.append(f"{root.name}\t(no_subfolders)\tN/A\tN/A")
        return rows
    for sub in sorted(subfolders, key=lambda p: p.name):
        txt_files = find_txt_files(sub)
        all_rewards_for_folder: List[float] = []
        if not txt_files:
            rows.append(f"{root.name}\t{sub.name}\tN/A\tN/A")
            continue
        for txt in sorted(txt_files, key=lambda p: p.name):
            try:
                rewards = parse_reward_values(txt)
            except Exception:
                rewards = []
            all_rewards_for_folder.extend(rewards)
        agg_mean, agg_var = compute_mean_variance(all_rewards_for_folder)
        rows.append(f"{root.name}\t{sub.name}\t{format_float_4(agg_mean)}\t{format_float_4(agg_var)}")
    return rows


def write_report(rows: List[str], output_path: Path) -> None:
    """将结果写入指定文件（覆盖写入）。"""
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for line in rows:
                f.write(line + os.linesep)
    except Exception as e:
        print(f"[ERROR] 写入报告失败: {output_path} ({type(e).__name__})", file=sys.stderr)


def build_pivot_rows(roots: List[Path]) -> List[str]:
    ds_names = [r.name for r in roots]
    data = {}
    for r in roots:
        ds = r.name
        data[ds] = {}
        subs = find_direct_subfolders(r)
        for sub in subs:
            txts = find_txt_files(sub)
            all_vals: List[float] = []
            for t in sorted(txts, key=lambda p: p.name):
                try:
                    rewards = parse_reward_values(t)
                except Exception:
                    rewards = []
                all_vals.extend(rewards)
            mean_val, var_val = compute_mean_variance(all_vals)
            data[ds][sub.name] = (mean_val, var_val)
    folders = set()
    for ds in ds_names:
        folders.update(list(data.get(ds, {}).keys()))
    folders = sorted(list(folders))
    header_cols = ["Folder"]
    for ds in ds_names:
        header_cols.append(f"{ds}_RewardMean")
        header_cols.append(f"{ds}_RewardVariance")
    rows: List[str] = ["\t".join(header_cols)]
    for fld in folders:
        cols = [fld]
        for ds in ds_names:
            mv = data.get(ds, {}).get(fld, (None, None))
            cols.append(format_float_4(mv[0]))
            cols.append(format_float_4(mv[1]))
        rows.append("\t".join(cols))
    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_roots = [
        # str(Path(r"d:/vscode/src_rtc/atla_test_results_huawei300kbps_delta")),
        str(Path(r"d:/vscode/src_rtc/gcc_test_results/")),
    ]
    parser = argparse.ArgumentParser(description="Reward statistics report generator")
    parser.add_argument(
        "--roots",
        type=str,
        nargs="*",
        default=default_roots,
        help="测试结果根目录列表（默认同时统计 atla_test_results 与 atla_test_results_rtc2_doubled）",
    )
    return parser.parse_args(argv)

def compute_worst_k_mean_variance(
    values: List[float], ratio: float = 0.1
) -> Tuple[Optional[float], Optional[float]]:
    """
    计算 reward 最差 ratio（默认 10%）的均值与总体方差（ddof=0）。
    若样本为空，返回 (None, None)。
    """
    if not values:
        return None, None

    n = len(values)
    k = max(1, int(n * ratio))

    try:
        if _HAS_NUMPY:
            arr = np.asarray(values, dtype=float)
            worst = np.sort(arr)[:k]
            mean_val = float(np.mean(worst))
            var_val = float(np.var(worst, ddof=0))
        else:
            vals = sorted(values)
            worst = vals[:k]
            mean_val = statistics.mean(worst)
            var_val = statistics.pvariance(worst)
        return mean_val, var_val
    except Exception:
        return None, None

def main(argv: Optional[List[str]] = None) -> int:
    # 日志文件放在脚本同目录
    script_dir = Path(__file__).resolve().parent
    log_path = script_dir / "reward_statistics_gcc.log"

    # 备份原始 stdout / stderr
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    # 启用 Tee
    sys.stdout = TeeLogger(log_path, orig_stdout)
    sys.stderr = TeeLogger(log_path, orig_stderr)

    try:
        args = parse_args(argv)
        roots = [Path(p) for p in args.roots]
        any_valid = False

        for root in roots:
            if not root.exists() or not root.is_dir():
                print(f"[ERROR] 根目录不存在或不可用: {str(root)}", file=sys.stderr)
                continue

            any_valid = True
            rows = build_table_rows(root)
            for line in rows:
                print(line)

        if not any_valid:
            return 2

        report_path = script_dir / "reward_statistics_report_gcc.txt"
        pivot_rows = build_pivot_rows(
            [r for r in roots if r.exists() and r.is_dir()]
        )
        write_report(pivot_rows, report_path)

        print(f"[INFO] 汇总报告已保存: {str(report_path)}")
        print(f"[INFO] 终端输出日志已保存: {str(log_path)}")

        return 0

    finally:
        # 恢复 stdout / stderr
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
    # args = parse_args(argv)
    # roots = [Path(p) for p in args.roots]
    # any_valid = False
    # for root in roots:
    #     if not root.exists() or not root.is_dir():
    #         print(f"[ERROR] 根目录不存在或不可用: {str(root)}", file=sys.stderr)
    #         continue
    #     any_valid = True
    #     rows = build_table_rows(root)
    #     for line in rows:
    #         print(line)
    # if not any_valid:
    #     return 2
    # script_dir = Path(__file__).resolve().parent
    # report_path = script_dir / "reward_statistics_report.txt"
    # pivot_rows = build_pivot_rows([r for r in roots if r.exists() and r.is_dir()])
    # write_report(pivot_rows, report_path)
    # print(f"[INFO] 汇总报告已保存: {str(report_path)}")
    # return 0


if __name__ == "__main__":
    sys.exit(main())

