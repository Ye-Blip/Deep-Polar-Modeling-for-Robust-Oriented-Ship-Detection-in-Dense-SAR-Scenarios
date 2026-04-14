import time
import torch
from ultralytics import YOLO


def validate_custom_model(model_path, data_path, imgsz=1024, batch=8):
    """
    高性能验证自定义旋转目标检测模型
    :param model_path: 模型权重文件路径 (.pt 或 .engine)
    :param data_path: 数据集配置文件路径 (.yaml)
    :param imgsz: 输入图像尺寸
    :param batch: 验证时的 batch size (显存 8G 建议 8-16, 显存 4G 建议 4)
    """

    # 1. 加载模型 (自动检测是否可以使用 GPU)
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path, task='obb')

    print(f"\n" + "=" * 40)
    print(f"开始验证模型: {model_path}")
    print(f"运行设备: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    print(f"输入尺寸: {imgsz} | Batch Size: {batch}")
    print("=" * 40)

    # 2. 运行验证 (加入优化参数)
    # half=True: 开启半精度，显存占用减半，速度翻倍
    # workers=4: 开启多线程数据预处理
    # rect=True: 矩形推理，减少不必要的边缘填充计算
    results = model.val(
        data=data_path,
        imgsz=imgsz,
        batch=batch,
        conf=0.001,
        iou=0.05,
        device=device,
        half=True,  # 【关键优化1】半精度推理
        workers=4,  # 【关键优化2】多线程加速数据读取
        rect=True,  # 【关键优化3】减少无效填充计算
        plots=False,  # 关闭绘图以节省保存时间
        save_json=False,
    )

    # 3. 提取核心指标
    # 注意：OBB 模型的指标通常以 (B) 结尾代表 Boundary Box
    precision = results.results_dict.get('metrics/precision(B)', 0)
    recall = results.results_dict.get('metrics/recall(B)', 0)
    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)

    # 4. 计算速度 (speed 包含 preprocess, inference, loss, postprocess)
    speed = results.speed
    inference_time = speed['inference']
    # 这里的 FPS 计算包含了端到端推理时间
    fps = 1000 / inference_time if inference_time > 0 else 0

    # 5. 打印易读的结果报告
    print("\n" + ">>> 验证指标结果 <<<")
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 30)
    print(f"{'Precision':<15} | {precision:.4f}")
    print(f"{'Recall':<15} | {recall:.4f}")
    print(f"{'mAP@.5':<15} | {map50:.4f}")
    print(f"{'mAP@.5:.95':<15} | {map50_95:.4f}")
    print("-" * 30)
    print(f"推理延迟 (ms/img): {inference_time:.2f}")
    print(f"每秒帧数 (FPS):    {fps:.2f}")
    print("=" * 40 + "\n")

    return results


if __name__ == "__main__":
    # --- 请根据你的实际路径修改 ---
    MODEL_WEIGHTS = r"C:\Users\29383\Desktop\ultralytics-main\my\sarm-2\weights\best.pt"
    DATA_CONFIG = r'C:\Users\29383\Desktop\ultralytics-main\my\data.yaml'

    # --- 提速建议 ---
    # 1. 如果还是慢，把 imgsz 改为 640
    # 2. 如果显存溢出 (Out of Memory)，把 batch 改为 4 或 2
    validate_custom_model(
        MODEL_WEIGHTS,
        DATA_CONFIG,
        imgsz=1024,
        batch=8  # 减小 batch 可以缓解内存压力导致的系统卡顿
    )