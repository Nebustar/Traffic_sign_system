import os
import sqlite3
import threading
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from Model.predict import detect_and_annotate

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ==========================================
# 考核点 2：手动实现基于队列的调度结构
# ==========================================
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

# 实例化全局任务队列和状态变量
image_queue = TaskQueue()
current_processing_task = None  # 记录当前正在识别的图片名称

# ==========================================
# 考核点 3：结果持久化 (SQLite 数据库)
# ==========================================
DB_PATH = 'history.db'

def init_db():
    """初始化数据库表"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            label TEXT,
            confidence TEXT,
            result_img_path TEXT,
            create_time DATETIME DEFAULT CURRENT_TIMESTAMP
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

# ==========================================
# 核心业务逻辑与后台调度
# ==========================================
'''
def mock_detect_and_annotate(input_path, filename):
    """模拟检测功能（与之前一致，略作精简）"""
    img = cv2.imread(input_path)
    h, w, _ = img.shape
    xmin, ymin = int(w * 0.25), int(h * 0.25)
    xmax, ymax = int(w * 0.75), int(h * 0.75)
    
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    label = "Speed Limit 50"
    confidence = "98.5%"
    cv2.putText(img, f"{label} {confidence}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    cv2.imwrite(output_path, img)
    return label, confidence, f"/{output_path}"

'''
def mock_detect_and_annotate(input_path,filename): # 模型适配，Line9 from Model.predict import detect_and_annotate
    output_path=os.path.join(app.config['OUTPUT_FOLDER'],filename)
    label,confidence=detect_and_annotate(input_path,output_path)
    return label,confidence,f"/{output_path}"

def queue_worker():
    """后台工作线程：不断检查队列，有任务就出队处理"""
    global current_processing_task
    while True:
        if not image_queue.is_empty():
            # 1. 任务出队
            task = image_queue.dequeue()
            filename = task['filename']
            filepath = task['filepath']
            
            # 2. 更新当前任务状态
            current_processing_task = filename
            
            # 模拟推理耗时 (让批处理过程在前端肉眼可见)
            time.sleep(2) 
            
            # 3. 执行推理
            label, conf, out_path = mock_detect_and_annotate(filepath, filename)
            
            # 4. 结果持久化入库
            save_to_db(filename, label, conf, out_path)
            
            # 5. 任务完成，清空当前状态
            current_processing_task = None
        else:
            time.sleep(1) # 队列为空时休息1秒

# 在 Flask 启动前，初始化数据库并开启后台调度线程
init_db()
worker_thread = threading.Thread(target=queue_worker, daemon=True)
worker_thread.start()

# ==========================================
# 路由接口 API
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "未选择文件"}), 400

    enqueued_count = 0
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
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

@app.route('/history', methods=['GET'])
def get_history():
    keyword = request.args.get('keyword', '')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, label, confidence, result_img_path, create_time FROM records WHERE filename LIKE ? OR label LIKE ? ORDER BY create_time DESC", 
              (f'%{keyword}%', f'%{keyword}%'))
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
    
    '''
    c.execute("DELETE FROM sqlite_sequence WHERE name = 'records'") # 清空序列计数器
    '''
    
    conn.commit()
    conn.close()
    
    return jsonify({"message": "已清空所有历史记录"}) 

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)