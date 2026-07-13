import os
import cv2
import sqlite3
import threading
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from Model.predict import detect_and_annotate
import re
import tempfile
import shutil
from moviepy import ImageSequenceClip

video_tasks = {}  # 记录视频任务状态

LABEL_ZH = {
    # 指示标志
    "i2": "非机动车行驶", "i4": "机动车行驶", "i5": "靠右侧道路行驶",
    "il100": "最低限速100km/h", "il60": "最低限速60km/h", "il80": "最低限速80km/h",
    "il90": "最低限速90km/h", "io": "其他指示", "ip": "停车让行",
    # 禁令标志
    "p10": "禁止机动车通行", "p11": "禁止鸣喇叭", "p12": "禁止电动自行车驶入",
    "p19": "禁止向右转弯", "p23": "禁止向左转弯", "p26": "禁止载货汽车驶入",
    "p27": "禁止运输危险物品车辆驶入", "p3": "禁止大型客车驶入", "p5": "禁止掉头",
    "p6": "禁止非机动车驶入", "pg": "减速让行", "ph4": "限高4米",
    "ph4.5": "限高4.5米", "ph5": "限高5米", "pl100": "限速100km/h",
    "pl120": "限速120km/h", "pl20": "限速20km/h", "pl30": "限速30km/h",
    "pl40": "限速40km/h", "pl5": "限速5km/h", "pl50": "限速50km/h",
    "pl60": "限速60km/h", "pl70": "限速70km/h", "pl80": "限速80km/h",
    "pm20": "限重20吨", "pm30": "限重30吨", "pm55": "限重55吨",
    "pn": "禁止停车", "pne": "禁止驶入", "po": "其他禁令", "pr40": "解除限速40",
    # 警告标志
    "w13": "十字交叉路口", "w32": "施工路段", "w55": "注意儿童",
    "w57": "注意行人", "w59": "前方合流", "wo": "其他警告标志",

    "no detection": "未检索到交通标识"
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500mb视频


class TaskQueue:
    def __init__(self):
        self.items = []  # 使用列表作为底层数据结构

    def enqueue(self, item):
        """入队：将新任务加入队列末尾"""
        self.items.append(item)

    def dequeue(self):
        """出队：取出队列最前端的任务 (先进先出 FIFO)"""
        if not self.is_empty():
            return self.items.pop(0)
        return None

    def is_empty(self):
        """队空判断"""
        return len(self.items) == 0

    def get_size(self):
        """获取队列当前长度"""
        return len(self.items)


image_queue = TaskQueue()
current_processing_task = None

DB_PATH = 'history.db'


def init_db():
    """初始化数据库表"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS records
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  filename
                  TEXT,
                  label
                  TEXT,
                  confidence
                  TEXT,
                  result_img_path
                  TEXT,
                  create_time
                  DATETIME
                  DEFAULT
                  CURRENT_TIMESTAMP
              )
              ''')
    conn.commit()
    conn.close()


def save_to_db(filename, label, confidence, result_path):
    """保存单次识别结果到数据库"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO records (filename, label, confidence, result_img_path) VALUES (?, ?, ?, ?)",
              (filename, label, confidence, result_path))
    conn.commit()
    conn.close()


def translate_labels(label_str):
    """辅助函数：将模型输出的类别代号串（如 'pl50, p11'）转换成中文描述串"""
    if not label_str or label_str.strip() == "no detection":
        return "未检测到交通标志"

    eng_labels = [l.strip() for l in label_str.split(',') if l.strip()]

    chn_labels = [LABEL_ZH.get(eng, eng) for eng in eng_labels]

    return ", ".join(chn_labels)


def mock_detect_and_annotate(input_path, filename):
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    label, confidence = detect_and_annotate(input_path, output_path)
    return label, confidence, f"/{output_path}"


def queue_worker():
    print("queue_worker 线程已启动！")
    """后台工作线程：不断检查队列，有任务就出队处理"""
    global current_processing_task
    while True:
        if not image_queue.is_empty():
            # 1. 任务出队
            task = image_queue.dequeue()

            if task.get("type") == "video":
                task_id = task["task_id"]
                video_idx = task["video_idx"]  # 第几帧
                input_path = task["input_path"]
                output_path = task["output_path"]
                # 第一帧切到推理阶段（拆帧已完成）
                if video_tasks[task_id].get("phase") == "uploading":
                    video_tasks[task_id]["phase"] = "processing"
                current_processing_task = f"{task_id} 视频 {video_idx}"
                try:
                    label_str, conf_str = detect_and_annotate(input_path, output_path)
                    if label_str and label_str != "no detection":
                        # 记录当前这一帧的识别标签
                        labels_list = [l.strip() for l in label_str.split(',') if l.strip()]
                        # 提取本帧置信度
                        confidences = []
                        try:
                            all_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", conf_str)
                            confidences = [float(n) for n in all_numbers]
                        except:
                            confidences = []

                        for idx, lab in enumerate(labels_list):
                            frame_conf = confidences[idx] / 100.0 if idx < len(confidences) else 0.0
                            if lab in video_tasks[task_id]['labels_conf']:
                                if frame_conf > video_tasks[task_id]['labels_conf'][lab]:
                                    video_tasks[task_id]['labels_conf'][lab] = frame_conf
                            else:
                                video_tasks[task_id]['labels_conf'][lab] = frame_conf
                                video_tasks[task_id]['labels_set'].add(lab)

                        try:
                            confs = []
                            all_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", conf_str)
                            for num_str in all_numbers:
                                confs.append(float(num_str))
                            if confs:
                                current_frame_max_conf = max(confs)
                                if current_frame_max_conf > video_tasks[task_id]['max_conf']:
                                    video_tasks[task_id]['max_conf'] = current_frame_max_conf
                        except Exception as parse_e:
                            print(f"置信度数值解析失败 (已忽略): {parse_e}")

                except Exception as e:
                    cv2.imwrite(output_path, cv2.imread(input_path))
                    print(f'视频帧 {video_idx} 处理失败：{e}')
                if task_id in video_tasks:
                    video_tasks[task_id]['processed_video'] += 1

                current_processing_task = None

                # 检查是不是每帧都处理完成
                if task_id in video_tasks:
                    info = video_tasks[task_id]
                    if info["processed_video"] >= info["total_video"]:
                        # ---- 进入合成阶段 ----
                        info["phase"] = "rendering"
                        info["render_progress"] = 0.0

                        video_dir = os.path.dirname(output_path)
                        out_files = sorted([f for f in os.listdir(video_dir) if f.startswith("out_")],
                                           key=lambda x: int(x.split('_')[1].split('.')[0]))  # 按帧排序
                        video_path = [os.path.join(video_dir, f) for f in out_files]
                        output_video_path = os.path.join(
                            app.config['OUTPUT_FOLDER'],
                            f"{task_id}.mp4"
                        )

                        # ---- 合成：优先使用 imageio 逐帧更新进度 ----
                        try:
                            import imageio
                            writer = imageio.get_writer(output_video_path, fps=info["fps"], codec='libx264')
                            total_frames = len(video_path)
                            for i, fpath in enumerate(video_path):
                                writer.append_data(imageio.imread(fpath))
                                if i % max(1, total_frames // 10) == 0 or i == total_frames - 1:
                                    info["render_progress"] = (i + 1) / total_frames
                            writer.close()
                            info["render_progress"] = 1.0
                        except ImportError:
                            # 回退到 moviepy
                            info["render_progress"] = 0.0
                            clip = ImageSequenceClip(video_path, fps=info["fps"])
                            clip.write_videofile(output_video_path, codec='libx264')
                            info["render_progress"] = 1.0

                        #  置信度过滤（>=0.8），0.8并不大，因为在整个视频中真实标签往往接近1，除非视频极其模糊
                        FILTER_CONF = 0.8
                        valid_labels = []
                        conf_pairs = []
                        for lab, conf in sorted(info['labels_conf'].items()):
                            if conf >= FILTER_CONF:
                                valid_labels.append(lab)
                                chinese_lab = LABEL_ZH.get(lab, lab)
                                conf_pairs.append(f"{chinese_lab}:{conf * 100:.1f}%")
                        translated_labels = [LABEL_ZH.get(lab, lab) for lab in valid_labels]
                        video_label = ", ".join(translated_labels) if translated_labels else "未检测到交通标志"
                        video_conf = ", ".join(conf_pairs) if conf_pairs else "0.0%"

                        original_name = info.get("original_filename", f"{task_id}.mp4")
                        save_to_db(
                            filename=original_name,
                            label=video_label,
                            confidence=video_conf,
                            result_path=f"/{output_video_path}"
                        )

                        info["status"] = "completed"
                        info["output_path"] = output_video_path

                        shutil.rmtree(video_dir)

            else:
                # 图片任务
                filename = task['filename']
                filepath = task['filepath']
                current_processing_task = filename
                try:
                    label, conf, out_path = mock_detect_and_annotate(filepath, filename)
                    chinese_label = translate_labels(label)
                    save_to_db(filename, chinese_label, conf, out_path)
                except Exception as e:
                    print(f"图片 {filename} 处理失败: {e}")
                finally:
                    current_processing_task = None
        else:
            time.sleep(1)


init_db()
worker_thread = threading.Thread(target=queue_worker, daemon=True)
worker_thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    """接收批量上传的文件并全部入队"""
    files = request.files.getlist('files')  # 获取多个文件
    if not files or files[0].filename == '':
        return jsonify({"error": "未选择文件"}), 400

    enqueued_count = 0
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 任务入队
            image_queue.enqueue({"filename": filename, "filepath": filepath})
            enqueued_count += 1

    return jsonify({"message": f"成功将 {enqueued_count} 张图片加入推理队列！"})


@app.route('/queue_status', methods=['GET'])
def queue_status():
    return jsonify({
        "queue_size": image_queue.get_size(),
        "current_task": current_processing_task,
        "is_idle": image_queue.is_empty() and current_processing_task is None
    })


@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "未选择视频"}), 400

    # 保存临时视频文件
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp_video.name)
    temp_video.close()

    task_id = f"video_{int(time.time())}"

    # 初始化视频任务状态（先不拆帧）
    video_tasks[task_id] = {
        "total_video": 0,
        "processed_video": 0,
        "status": "processing",
        "phase": "uploading",
        "upload_progress": 0.0,
        "render_progress": 0.0,
        "output_path": None,
        "fps": 0,
        "labels_set": set(),
        "labels_conf": {},
        "max_conf": 0.0,
        "original_filename": video_file.filename
    }

    # 启动拆帧线程
    def split_video():
        try:
            cap = cv2.VideoCapture(temp_video.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_videos = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_tasks[task_id]["total_video"] = total_videos
            video_tasks[task_id]["fps"] = fps

            videos_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
            os.makedirs(videos_dir, exist_ok=True)

            for idx in range(total_videos):
                ret, frame = cap.read()
                if not ret:
                    break
                video_path = os.path.join(videos_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(video_path, frame)
                # 推理任务入队
                image_queue.enqueue({
                    "type": "video",
                    "task_id": task_id,
                    "video_idx": idx,
                    "input_path": video_path,
                    "output_path": os.path.join(videos_dir, f"out_{idx:06d}.jpg")
                })
                # 更新拆帧进度
                if idx % max(1, total_videos // 10) == 0 or idx == total_videos - 1:
                    video_tasks[task_id]["upload_progress"] = (idx + 1) / total_videos

            cap.release()
            os.unlink(temp_video.name)
            # 拆帧完成，切换到推理阶段
            video_tasks[task_id]["phase"] = "processing"
        except Exception as e:
            print(f"拆帧失败 {task_id}: {e}")
            video_tasks[task_id]["status"] = "error"

    threading.Thread(target=split_video, daemon=True).start()

    return jsonify({
        "message": "视频已开始拆帧，请查看进度",
        "task_id": task_id,
        "total_frames": 0,
        "original_filename": video_file.filename
    })


@app.route('/video_status/<task_id>', methods=['GET'])
def video_status(task_id):
    if task_id not in video_tasks:
        return jsonify({"error": "任务不存在"}), 404
    info = video_tasks[task_id]

    # 根据阶段计算综合进度
    phase = info.get("phase", "uploading")
    if phase == "uploading":
        progress = info.get("upload_progress", 0.0) * 33
    elif phase == "processing":
        frame_progress = info["processed_video"] / info["total_video"] if info["total_video"] > 0 else 0
        progress = 33 + frame_progress * 33
    elif phase == "rendering":
        render_progress = info.get("render_progress", 0.0)
        progress = 66 + render_progress * 33
    else:
        progress = 100
    progress = min(progress, 100)

    return jsonify({
        "task_id": task_id,
        "total_videos": info["total_video"],
        "processed_videos": info["processed_video"],
        "progress": progress,
        "phase": phase,
        "status": info["status"],
        "output_path": info["output_path"],
        "original_filename": info["original_filename"]
    })


# 考核点 4：查找历史记录
@app.route('/history', methods=['GET'])
def get_history():
    """根据关键字模糊查询历史记录"""
    keyword = request.args.get('keyword', '')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
              SELECT id, filename, label, confidence, result_img_path, create_time
              FROM records
              WHERE id = ?
                 OR filename LIKE ?
                 OR label LIKE ?
              ORDER BY id DESC
              """,
              (keyword, f'%{keyword}%', f'%{keyword}%')
              )
    rows = c.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row[0], "filename": row[1], "label": row[2],
            "confidence": row[3], "result_img_path": row[4], "time": row[5]
        })
    return jsonify(results)


@app.route('/history/<int:record_id>', methods=['DELETE'])
def delete_single_history(record_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("SELECT result_img_path FROM records WHERE id = ?", (record_id,))
        row = c.fetchone()

        if row and row[0]:
            filepath = row[0].lstrip('/')
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as file_err:
                    print(f"硬盘文件删除失败: {file_err}")

        c.execute("DELETE FROM records WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "message": f"记录 {record_id} 已成功删除"})

    except Exception as e:
        print(f"后端删除接口发生致命崩溃: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/history/all', methods=['DELETE'])
def delete_all_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for filename in os.listdir(app.config['OUTPUT_FOLDER']):
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.isfile(filepath):
            os.remove(filepath)

    c.execute("DELETE FROM records")
    c.execute("DELETE FROM sqlite_sequence WHERE name = 'records'")

    conn.commit()
    conn.close()

    return jsonify({"message": "已清空所有历史记录"})


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
