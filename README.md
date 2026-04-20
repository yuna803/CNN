# CNN

实验实验作业：基于 TensorFlow /Pytorch的卷积神经网络实现与优化
一、实验主题
基于 TensorFlow/Pytorch 框架搭建卷积神经网络（CNN），完成 CIFAR-10 数据集的分类任务，实现数据预处理、模型构建、训练优化全流程，对比不同优化策略（Mini-batchSGD、Momentum、RMSprop、Adam）的收敛性能与泛化效果，深入理解 CNN 的特征提取机制及优化算法的作用原理。
二、核心涵盖知识点
1.卷积神经网络基础：TensorFlow/Pytorch 中卷积层（Conv2D）、池化层（MaxPooling2D）、全连接层（Dense）的搭建；
2.数据预处理：CIFAR-10 数据集加载、归一化、独热编码、数据集划分及 Mini-batch 生成；
3.损失函数与优化器：交叉熵损失的调用与适配、TensorFlow 内置 / 自定义优化器的使用；
4.模型训练与验证：训练循环构建、早停（EarlyStopping）、模型保存与加载；
5.优化算法对比：Mini-batchSGD、Momentum、RMSprop、Adam 的参数配置与性能差异；
6.模型评估与可视化：准确率计算、损失曲线 / 准确率曲线绘制、混淆矩阵可视化；
7.模块化编程：自定义函数封装、TensorFlow 高阶 API（如 tf.data）的使用。

三、实验核心要求
1. 无框架限制：可选使用TensorFlow、PyTorch等深度学习框架，也可基于NumPy手动实现(答辩时自己能理解就行)；
2. 函数封装要求：
(1) 自定义封装函数不少于3个(需涵盖数据处理、模型核心逻辑、结果 分析等不同模块)；
(2) 函数需具备明确功能、参数说明、返回值定义，体现模块设计思想；
(3) 必须在主函数中调用自定义封装的函数不少于3个，且需体现函数 间的嵌套/依赖调用(如A函数调用B函数，主函数调用A函数)；
3. 网络结构要求：采用4-6层全连接神经网络(输入层+3-5个隐藏层+输出层)，每个隐藏层神经元数量需合理设计(如1024、512、256、128等，需说明设计依据)；
4. 结果可视化与分析：完成损失曲线、性能对比图、混淆矩阵等绘制，结合理论知识解释实验差异。
1.框架限制：基于 TensorFlow 2.x（推荐 2.8+）实现，合理使用 tf.keras 高阶 API，禁止直接调用第三方封装的完整 CNN 模型；
2.函数封装要求：
(1) 自定义封装函数不少于 5 个（涵盖数据处理、模型构建、训练、评估、可视化等模块）；
(2) 每个函数需标注清晰的功能说明、参数列表、返回值定义；
(3) 主函数中需调用不少于 5 个自定义函数，且体现函数间嵌套依赖（如模型构建函数被训练函数调用，训练函数被主函数调用）；
3.网络结构要求：
(1) 卷积神经网络结构：至少包含 2 组 “卷积层 + 池化层”（Conv2D+MaxPooling2D）+ 1-2 个全连接层；
(2) 卷积层参数：卷积核数量（如 32、64、128）、大小（3×3/5×5）、步幅、填充方式需合理设计，并说明设计依据；
(3) 激活函数：隐藏层使用 ReLU，输出层使用 Softmax；
4.结果可视化与分析：
(1) 绘制 4 种优化算法的训练 / 验证损失曲线（同一图）、训练 / 验证准确率曲线（同一图）；
(2) 绘制最优模型在测试集上的混淆矩阵；
(3) 结合理论分析不同优化算法的收敛速度、最终性能差异及原因。

四、实验任务
(一)数据集与预处理(需封装相关函数)
1. 选用CIFAR-10数据集，基于纯Python/NumPy完成数据加载(解压、二进制解析)；
2. 封装函数1：data_preprocess(x_raw,y_raw,val_ratio=0.1,test_ratio=0.1)
(1) 功能：整合“归一化、独热编码、数据集划分”功能，输入原始数据与划分比例，返回处理后的训练集(x_train,y_train)、验证集(x_val,y_val)、测试集(x_test,y_test)；
(2) 要求：函数内部需调用至少1个自定义子函数(如one_hot_encode(y)专门实现独热编码)；
3. 封装函数2：create_mini_batches(x,y,batch_size=64,shuffle=True)
(1) 功能：输入处理后的数据集，生成Mini-batch迭代器，支持批次大小自定义和数据打乱；
(2) 要求：返回可迭代的批次数据(每次返回一个批次的特征x_batch和 标签y_batch)。
·  数据集加载：使用 TensorFlow 内置接口加载 CIFAR-10 数据集；
封装函数 1：data_preprocess(val_ratio=0.1, test_ratio=0.1)
(1)功能：整合 “数据加载、归一化、独热编码、数据集划分” 功能；
(2) 参数：val_ratio（验证集比例）、test_ratio（测试集比例）；
(3) 返回值：(x_train, y_train), (x_val, y_val), (x_test, y_test)（均为 TensorFlow 张量）；
(4) 要求：内部调用自定义子函数one_hot_encode(y, num_classes=10)实现独热编码。
封装函数 2：create_mini_batches(x, y, batch_size=64, shuffle=True)
(1)功能：基于 tf.data.Dataset 生成 Mini-batch 迭代器；
(2) 参数：特征x、标签y、批次大小batch_size、是否打乱shuffle；
(3) 返回值：tf.data.Dataset 对象（可迭代的批次数据）；
(4) 要求：支持数据打乱、批次划分，适配 TensorFlow 训练流程。

(二)神经网络基础模块实现(需封装核心函数)
1. 封装子函数：relu(z)、relu_derivative(z)、softmax(z)
(1) 功能：分别实现ReLU激活函数、ReLU导数、Softmax激活函数(Softmax需处理数值溢出问题)；
2. 封装函数3：initialize_parameters(layer_dims)
(1) 功能：输入网络各层维度(如[784,1024,512,256,10]，对应输入层→3个隐藏层→输出层)，基于Xavier或He初始化方法，返回初始化后的权重(W1,W2,...,Wn)和偏置(b1,b2,...,bn)；
3. 封装函数4：forward_propagation(x,params)
(1) 功能：输入特征数据x和模型参数params(含所有权重和偏置)，实现多层网络的前向传播流程；
(2) 要求：返回最终预测概率y_pred(Softmax输出)和各层中间输出(z1,a1,z2,a2,...,zn-1,an-1，用于反向传播计算)；
(3) 依赖：函数内部需调用relu()和softmax()子函数。

封装函数 3：build_cnn_model(input_shape=(32,32,3), num_classes=10)
(1)功能：构建基于 TensorFlow 的 CNN 模型；
(2) 参数：输入形状input_shape（CIFAR-10 为 32×32×3）、类别数num_classes；
(3) 返回值：tf.keras.Model 对象（编译前的模型）；
(4) 要求：
模型结构示例（可自定义）：Input 层 → Conv2D (32, 3, activation='relu') → MaxPooling2D (2) → Conv2D (64, 3, activation='relu') → MaxPooling2D (2) → Flatten → Dense (128, activation='relu') → Dense (num_classes, activation='softmax')；
(需注释各层作用及参数设计依据。)
封装函数 4：compile_model(model, optimizer='adam', learning_rate=0.001)
(1)功能：编译 CNN 模型，指定优化器、损失函数、评估指标；
(2) 参数：model（构建好的 CNN 模型）、optimizer（优化器名称，支持 sgd、momentum、rmsprop、adam）、learning_rate（学习率）；
(3) 返回值：编译后的 tf.keras.Model 对象；
(4) 要求：针对不同优化器配置对应参数（如 Momentum 的 momentum=0.9，RMSprop 的 rho=0.9）。

(三)损失函数与反向传播实现
封装函数 5：train_model(model, train_ds, val_ds, epochs=50, callbacks=None)
(1)功能：执行模型训练流程，记录训练 / 验证损失和准确率；
(2) 参数：model（编译后的模型）、train_ds（训练集 Mini-batch 迭代器）、val_ds（验证集 Mini-batch 迭代器）、epochs（训练轮数）、callbacks（回调函数列表，如早停）；
(3) 返回值：训练历史对象（tf.keras.callbacks.History）、训练好的模型；
(4) 依赖：内部需调用create_mini_batches生成验证集批次，支持早停回调防止过拟合。
封装函数 6：train_with_different_optimizers(build_fn, train_data, val_data, test_data, optimizers_dict, epochs=50)
(1)功能：循环使用不同优化器训练模型，记录各优化器的训练结果；
(2) 参数：build_fn（模型构建函数）、train_data/val_data/test_data（训练 / 验证 / 测试集）、optimizers_dict（优化器字典，key 为优化器名称，value 为优化器配置）、epochs；
(3) 返回值：结果字典（key 为优化器名称，value 包含训练历史、最终模型）；
(4) 要求：支持同时训练 Mini-batchSGD、Momentum、RMSprop、Adam 四种优化器，保证其他参数（学习率、epochs 等）一致。
(四) 结果可视化与模型评估（需封装核心函数）
封装函数 7：plot_performance(results_dict)
(1)功能：可视化不同优化器的训练 / 验证损失、准确率曲线；
(2) 参数：results_dict（不同优化器的训练结果字典）；
(3) 输出：生成 2 幅对比图 ——① 损失曲线（x 轴为 epochs，y 轴为损失值，不同颜色代表不同优化器，区分训练 / 验证）；② 准确率曲线（同 x 轴，y 轴为准确率）；
(4) 要求：图表需标注标题、坐标轴、图例，样式清晰易读。
封装函数 8：evaluate_model(model, test_ds, class_names)
(1)功能：评估模型在测试集上的性能，计算准确率并绘制混淆矩阵；
(2) 参数：model（训练好的模型）、test_ds（测试集 Mini-batch 迭代器）、class_names（类别名称列表）；
(3) 返回值：测试集准确率；
(4) 要求：使用 seaborn 绘制混淆矩阵热力图，标注类别名称，显示各类别预测准确率。

(五)结果可视化与模型评估
封装函数8：plot_performance(results_dict)
(1) 功能：输入不同优化算法的训练结果字典(key为算法名称，value为train_losses、val_losses、train_accs、val_accs)；
(2) 要求：绘制2幅对比图——①不同算法的训练/验证损失曲线(同一图)；②不同算法的训练/验证准确率曲线(同一图)；
封装函数9：evaluate_model(params,x_test,y_test)
(1) 功能：输入最优模型参数params、测试集特征x_test、测试集标签y_test，计算测试集准确率并绘制混淆矩阵(分类任务)；
(2) 依赖：函数内部需调用forward_propagation函数。


(六)主函数设计(核心整合)
编写主函数main()，完成以下完整流程，且必须调用不少于 3个自定义封装函数：
1.数据准备：调用data_preprocess加载并预处理 CIFAR-10 数据集，调用create_mini_batches生成训练 / 验证 / 测试集 Mini-batch 迭代器；
2.模型构建：定义优化器字典（包含 sgd、momentum、rmsprop、adam），调用build_cnn_model构建 CNN 模型结构；
3.模型训练：调用train_with_different_optimizers使用 4 种优化器分别训练模型，记录训练结果；
4.结果可视化：调用plot_performance绘制损失 / 准确率对比曲线；
5.最优模型评估：选择验证集准确率最高的模型，调用evaluate_model在测试集上评估性能，输出测试准确率并绘制混淆矩阵；
6.结果分析：打印不同优化器的收敛速度、最终准确率对比，简要分析差异原因。

五、实验报告要求
1.实验目的：明确基于 TensorFlow 实现 CNN 的目标、核心知识点及对比不同优化算法的意义；
2.实验环境：列出硬件配置（CPU/GPU 型号）、软件版本（Python、TensorFlow、NumPy、Matplotlib 等）；
3.数据集介绍与预处理：(1) 描述 CIFAR-10 数据集（样本数量、尺寸、类别、数据分布）；(2) 说明数据预处理步骤（归一化、独热编码、数据集划分）的目的及实现逻辑；(3) 解释 tf.data.Dataset 生成 Mini-batch 的优势；
4.模型设计：(1) 详细说明 CNN 结构（各层类型、参数、数量）及设计依据（如卷积核大小选择、通道数递增的原因）；(2) 解释激活函数、损失函数的选择依据；
5.核心函数实现：(1) 重点阐述build_cnn_model、train_with_different_optimizers的实现逻辑，附关键代码片段；(2) 说明 4 种优化器的参数配置及 TensorFlow 中的调用方式；
6.实验结果与分析：(1) 展示 4 种优化算法的损失曲线、准确率曲线及最优模型的混淆矩阵；(2) 分析不同优化算法的收敛速度、最终性能差异（如 Adam 收敛最快、SGD 收敛最慢但泛化性较好）；(3) 讨论模型的泛化能力（训练集与验证集性能差异）、过拟合现象及改进方向（如添加 Dropout、L2 正则化）；
7.函数封装说明：(1) 列出所有自定义函数的功能、参数、返回值；(2) 绘制函数间调用流程图（如 main→train_with_different_optimizers→compile_model→build_cnn_model）；
8.结论与收获：总结实验的主要发现、TensorFlow 框架的使用心得、遇到的问题及解决方案（如梯度消失、过拟合、训练速度慢等）。
