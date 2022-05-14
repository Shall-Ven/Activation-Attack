import json

def wr_json(data: dict = None,  # 字典格式数据
            path_file: str = None,  # 数据保存路径
            type: str = None  # 读写方式
            ):
    if type == '写入':
        dict_json = json.dumps(data)  # 转化为json格式文件
        with open(path_file, 'w+') as file:
            file.write(dict_json)
    elif type == '读取':
        # 读取.json格式文件的内容
        with open(path_file, 'r+') as file:
            content = file.read()
            content = json.loads(content)  # 将json格式文件转化为python的字典文件
        return content
    elif type == '修改添加':  # 同键修改，异键添加
        # 读取.json格式文件的内容
        with open(path_file, 'r+') as file:
            content = file.read()
            content = json.loads(content)  # 将json格式文件转化为python的字典文件

        for key in data.keys():
            content[key] = data[key]

        content = json.dumps(content)  # 转化为json格式文件
        with open(path_file, 'w+') as file:
            file.write(content)


def predict_image(model, image):
    output = model(image)
    output = torch.argmax(output, dim=1).item()
    return output
