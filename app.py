import logging
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, make_response, send_file
import os
import tempfile
import sys
sys.path.append(r"D:\\vscode\\local_model\\granite-tsfm-main\\granite-tsfm-main")  # 如果需要，添加test3.py所在目录
import seq_pred

app = Flask(__name__)
# 设置上传文件的临时存储路径
# app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
# app.config['UPLOAD_FOLDER'] = "D:\Code\code_NLP课设\前端\file"
app.config['UPLOAD_FOLDER'] = r"D:\\vscode\\local_model\\file"



app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024        # 16MB文件大小限制




def process_csv(file_path):
    """自定义CSV处理逻辑"""
    # 读取CSV文件


    seq_pred.zeroshot_eval(dataset_name='temp',context_length=512, forecast_length=96, batch_size=64)
   ###########################################################

    img_path = "img.png"


    return img_path


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    # logging.info("1")  # 使用标准日志代替 print
    # print("------------------------")

    # GET请求
    if request.method == 'GET':
        return render_template('index.html', result=result)

    # POST请求
    # 检查是否有文件上传
    if 'csv_file' not in request.files:
        result = "请选择CSV文件"
        return render_template('index.html', result=result)

    file = request.files['csv_file']
    if file.filename == '':
        result = "请选择有效的CSV文件"
        return render_template('index.html', result=result)

    # 确保文件是CSV格式
    if not file.filename.endswith('.csv'):
        result = "请上传CSV格式的文件"
        return render_template('index.html', result=result)

    # try:
    # 保存上传的文件到临时位置
    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.csv')
    file.save(temp_file)
    # print("")

    img_path  = process_csv(temp_file)



    # # 处理CSV文件
    # # 生成处理后的文件名
    # output_file = os.path.join(app.config['UPLOAD_FOLDER'],
    #                            f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")

    # # 保存处理后的文件
    # processed_df.to_csv(output_file, index=False)

    # # # 提供文件下载
    # response = make_response(send_file(output_file, as_attachment=True))
    # response.headers["Content-Disposition"] = f"attachment; filename={os.path.basename(output_file)}"

    # # 处理完成后删除临时文件
    # os.remove(temp_file)
    # os.remove(output_file)
    # response = make_response()

    # return response

    return render_template('index.html',img_path=img_path)


    # except Exception as e:
    #     result = f"处理文件时出错: {str(e)}"

@app.route('/img', methods=['GET'])
def img():
    return render_template('index.html', img_path="img.png")

if __name__ == '__main__':
    app.run(debug=True)