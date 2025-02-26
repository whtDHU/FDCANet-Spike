import os
import re

import numpy as np

# 定义日志文件路径模板
K = 4
log_dir = f"/e/wht_project/eeg_data/k_fold/K={K}_channel_models/"
log_file_template = "Patient_{n1}_Fold_{n2}_K_{K}.log"

# 正则表达式匹配日志中的指标
metrics_pattern = {
    'accuracy': re.compile(r'Best validation accuracy:\s+(\d+\.\d+)'),
    'sensitivity': re.compile(r'sensitivity/recall:\s+(\d+\.\d+)%'),
    'specificity': re.compile(r'specificity:\s+(\d+\.\d+)%'),
    'precision': re.compile(r'precision:\s+(\d+\.\d+)%'),
    'F1_score': re.compile(r'F1_score:\s+(\d+\.\d+)%')
}

# 定义统计指标
metrics_summary = {
    'accuracy': [],
    'sensitivity': [],
    'specificity': [],
    'precision': [],
    'F1_score': []
}

# 存储每个病人的结果
patient_results = {}


# 解析日志文件函数
def parse_log_file(log_file_path):
    results = {}
    with open(log_file_path, 'r') as f:
        content = f.read()
        for metric, pattern in metrics_pattern.items():
            match = pattern.search(content)
            if match:
                results[metric] = float(match.group(1))
    return results


# 累积所有病人的指标总和
total_metrics = {metric: 0 for metric in metrics_pattern}
valid_patient_count = 0

# 遍历每个病人的日志文件
for patient_id in range(1, 22):  # 病人编号从1到21
    patient_metrics = {metric: [] for metric in metrics_pattern}

    for fold in range(10):  # 10折交叉验证
        log_file_path = os.path.join(log_dir, log_file_template.format(n1=patient_id, n2=fold,K=K))
        if os.path.exists(log_file_path):
            fold_results = parse_log_file(log_file_path)
            for metric, value in fold_results.items():
                patient_metrics[metric].append(value)

    # 计算每个病人的十折平均结果
    if all(patient_metrics[metric] for metric in metrics_pattern):  # 确保所有指标都有值
        valid_patient_count += 1
        patient_avg_metrics = {metric: np.mean(values) for metric, values in patient_metrics.items() if values}
        patient_results[patient_id] = patient_avg_metrics

        # 将每个病人的平均值累加到 total_metrics
        for metric, avg_value in patient_avg_metrics.items():
            total_metrics[metric] += avg_value

# 输出每个病人的平均结果
for patient_id, metrics in patient_results.items():
    print(f"Patient {patient_id}:")
    for metric, avg_value in metrics.items():
        print(f"  {metric}: {avg_value:.2f}")

# 计算所有病人的平均值
overall_avg_metrics = {metric: total / valid_patient_count for metric, total in total_metrics.items()}

print("\nOverall average metrics for all patients:")
for metric, avg_value in overall_avg_metrics.items():
    print(f"  {metric}: {avg_value:.2f}")