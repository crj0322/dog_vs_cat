# 猫狗大战

## 数据
[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

## 项目要求
https://github.com/nd009/capstone/tree/master/dog_vs_cat

## 文件目录
### classification: 分类代码  
- data_preprocess.ipynb ------------------------------------ 数据预处理（异常值、裁剪）  
- ref_data.ipynb ------------------------------------------- 生成映射目录  
- gen_feature.ipynb ---------------------------------------- 生成、保存特征向量  
- single_model.ipynb --------------------------------------- 单模型实验、分类器预训练  
- single_model_fine_tune.ipynb ----------------------------- 单模型卷积层fine-tune  
- multi_model.ipynb ---------------------------------------- 多模型融合分类器实验  
- class_activation_map.ipynb ------------------------------- 生成特征图  
- outliers_total.json -------------------------------------- 异常值
  
### docs: 文档  
- proposal.pdf --------------------------------------------- 开题报告  
- proposal_data.ipynb -------------------------------------- 数据可视化  
- proposal_sketch.docx ------------------------------------- 开题报告草稿  
- plot_model.ipynb ----------------------------------------- 模型可视化
- report_sketch.docx --------------------------------------- 项目报告草稿
- report.pdf ----------------------------------------------- 项目报告
- data_preprocess.html ------------------------------------- 项目代码
- ref_data.html -------------------------------------------- 项目代码
- gen_feature.html ----------------------------------------- 项目代码
- single_model.html ---------------------------------------- 项目代码
- single_model_fine_tune.html ------------------------------ 项目代码
- multi_model.html ----------------------------------------- 项目代码
- class_activation_map.html -------------------------------- 项目代码

## 项目环境
- Ubuntu 16.04 LTS
- NVIDIA GTX 1080
- CUDA 9.0
- cuDNN 7.0
- Anaconda 3.5 (Python 3.6)
- Tensorflow 1.6
- Keras 2.1.5

## 项目流程
1. ref_data 映射数据；
2. data_preprocess 数据清洗，使用三种模型预测异常值，每个模型约15分钟，产生 outliers_total；
3. ref_data 重映射、分割数据；
4. gen_feature 生成特征向量，每个模型约10分钟；
5. single_model 分别实验 InceptionV3，Xception，InceptionResNetV2；
6. single_model_fine_tune 微调 Xception 卷积层，每epoch约460s，共20个epochs约2.5小时；
7. multi_model 模型融合；
8. class_activation_map 生成特征图。

![cam1](https://github.com/crj0322/dog_vs_cat/raw/master/docs/cam1.png)
![cam2](https://github.com/crj0322/dog_vs_cat/raw/master/docs/cam2.png)
![cam3](https://github.com/crj0322/dog_vs_cat/raw/master/docs/cam3.png)

## Detection
### yolov3: Detection codes  
- yolo_model.ipynb -------------------------------------------- build yolov3 model and predict bboxes, steal codes from [HERE](https://github.com/qqwweee/keras-yolo3).
- todo: train yolo with dog cat data.
