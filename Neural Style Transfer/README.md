# 神经风格迁移

## 环境配置
python==3.7.5 torch==1.8.1 torchversion==0.9.1

## 使用方法

1. 将content image存入路径'./images/content_image/',将style image存入路径'./images/style_image';
2. 运行脚本'main.py'：
>\>\>python main.py
1. 根据提示，输入对应加后缀的文件名，以及合成文件的文件名，等待训练即可，合成的图片会自动保存在'./outputs'目录中。

## 运行参考
1. 将content image起名为'content.jpg'存放在'./images/content_image/'，将style image起名为'style_image.jpg'存放在'./images/style_image/';
2. 运行脚本'main.py'，并根据提示进行操作:
```
>> python main.py
>> Loading images...
>> Please input content image file name: content_image.jpg
>> Please input style image file name: style_image.jpg
>> Please input output file name: output.jpg
>> Creating model...
>> Training...
>> ...
>> Image saved to './outputs/output.jpg'
```
3. 从目录'./outputs'中查看结果

![结果]('./outputs/output.jpg')

## 参考文献
[Leon A. Gatys, Alexander S. Ecker, Matthias Bethge. A Neural Algorithm of Artistic Style. 2015,arXiv:1508.06576.](https://arxiv.org/abs/1508.06576)