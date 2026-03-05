import json
import random
import os


def make_mini_json_by_ratio(input_path, output_path, ratio=0.2, seed=42):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return

    # 读取原始 JSON 数据
    with open(input_path, 'r') as f:
        data = json.load(f)

    # 获取所有的样本键值
    all_keys = list(data.keys())
    total_size = len(all_keys)

    # 动态计算 20% 的样本数量 (向下取整)
    keep_size = int(total_size * ratio)

    if keep_size == 0:
        print(f"警告: 20% 的数据量为 0，请检查源文件 {input_path} 是否为空。")
        return

    # 关键步骤：设置固定的随机种子，保证每次运行抽出的 20% 是同一批
    random.seed(seed)

    # 随机抽取指定数量的键值
    mini_keys = random.sample(all_keys, keep_size)

    # 根据抽取到的键值构建新的迷你字典
    mini_data = {k: data[k] for k in mini_keys}

    # 将截取后的数据保存为新的 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(mini_data, f, indent=4)

    print(f"原数据共 {total_size} 条，成功抽取 {ratio * 100}% (共 {keep_size} 条) -> 保存至 {output_path}")


if __name__ == '__main__':
    # 设置你要抽取的比例，0.2 代表 20%
    target_ratio = 0.20

    # 处理训练集
    make_mini_json_by_ratio(
        input_path='data_file/MAVL/rad_graph_metric_train.json',
        output_path='data_file/MAVL/rad_graph_metric_train_mini.json',
        ratio=target_ratio,
        seed=42
    )

    # 处理验证集
    make_mini_json_by_ratio(
        input_path='data_file/MAVL/rad_graph_metric_validate.json',
        output_path='data_file/MAVL/rad_graph_metric_validate_mini.json',
        ratio=target_ratio,
        seed=42
    )

    # 处理测试集
    make_mini_json_by_ratio(
        input_path='data_file/MAVL/rad_graph_metric_test.json',
        output_path='data_file/MAVL/rad_graph_metric_test_mini.json',
        ratio=target_ratio,
        seed=42
    )