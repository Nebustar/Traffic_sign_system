import sqlite3
import pandas as pd
from pathlib import Path


# =========================
# 路径配置
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "history.db"

OUTPUT_PATH = BASE_DIR / "history_export.xlsx"



# =========================
# 导出Excel函数
# =========================

def export_excel():

    # 判断数据库是否存在

    if not DATABASE_PATH.exists():

        raise FileNotFoundError(
            "未找到history.db，请先运行系统完成一次识别任务。"
        )


    # 连接数据库

    conn = sqlite3.connect(
        DATABASE_PATH
    )


    # 查询所有历史记录

    sql = """
    SELECT
        id,
        filename,
        label,
        confidence,
        result_img_path,
        create_time
    FROM records
    ORDER BY id ASC
    """


    # 读取数据库

    data = pd.read_sql_query(
        sql,
        conn
    )


    conn.close()



    # 写入Excel

    data.to_excel(
        OUTPUT_PATH,
        index=False,
        engine="openpyxl"
    )


    print("==============================")
    print("Excel导出完成！")
    print(f"保存位置：{OUTPUT_PATH}")
    print(f"导出数据量：{len(data)} 条")
    print("==============================")



# =========================
# 主程序入口
# =========================

if __name__ == "__main__":

    export_excel()