from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from task_queue import TaskQueue
from hust_yolo.evaluate import add_eval_args, run_eval
from hust_yolo.predict import add_predict_args, run_predict
from hust_yolo.prepare_cctsdb import add_prepare_cctsdb_args, run_prepare_cctsdb
from hust_yolo.prepare_dataset import add_prepare_args, run_prepare
from hust_yolo.train import add_train_args, run_train
from task_queue import TaskQueue

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HUST YOLO baseline command line")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build YOLO train/val/test dataset")
    add_prepare_args(prepare_parser)

    prepare_cctsdb_parser = subparsers.add_parser("prepare-cctsdb", help="Convert CCTSDB to YOLO train/val/test")
    add_prepare_cctsdb_args(prepare_cctsdb_parser)

    train_parser = subparsers.add_parser("train", help="Train a YOLO baseline")
    add_train_args(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained YOLO model")
    add_eval_args(eval_parser)

    predict_parser = subparsers.add_parser("predict", help="Run YOLO inference")
    add_predict_args(predict_parser)

    batch_parser = subparsers.add_parser("batch", help="Batch inference with queue")
    batch_parser.add_argument("images", nargs="+", help="One or more image paths")
    # 可选添加其他参数，例如：
    batch_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    batch_parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    if args.command == "prepare":
        run_prepare(args, project_root)
    elif args.command == "prepare-cctsdb":
        run_prepare_cctsdb(args, project_root)
    elif args.command == "train":
        run_train(args, project_root)
    elif args.command == "eval":
        run_eval(args, project_root)
    elif args.command == "predict":
        run_predict(args, project_root)
    else:
        parser.error(f"Unsupported command: {args.command}")


def run_batch(args: argparse.Namespace, project_root: Path) -> None:
    import cv2
    import numpy as np


    from manualDataProcessing import manual_processing

    queue = TaskQueue()
    count=0

    for i in args.images:
        pic_path = Path(i)
        if not pic_path.exists():
            print(f"{pic_path} 不存在。\n")
            continue

        queue.enqueue(pic_path)
        count+=1
        print(f"{pic_path}入队。\n")

    if queue.is_empty():
        print(f"没有有效图片，程序结束。\n")
        return
    print(f"{queue.size}张图片待处理。\n")

    import onnxruntime as ort
    model_path='模型路径，待填'
    if not Path(model_path).exists():
        print("模型不存在")
        return
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print(f"模型加载成功: {model_path}\n")

    for i in range(queue.size):
        pic_path = queue.dequeue()
        try:
            pic_pil=Image.open(pic_path)
            pic_rgb=np.array(pic_pil)

            input_tensor=manual_processing(pic_rgb,(args.imgsz, args.imgsz))
            input_batch = np.expand_dims(input_tensor, axis=0)


            outputs = session.run(None, {input_name: input_batch})

            print(f"图片{pic_path}处理完成")
            print(f"输出形状: {[out.shape for out in outputs]}")
        except Exception as e:
            print(f"图片{pic_path}处理出错 {pic_path}: {e}")
    print(f"全部处理完成")
if __name__ == "__main__":
    main()



