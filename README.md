## Chinese Entity Recognize 2018-10

#### 1.preprocess

general_prepare() 将每行数据分割为 (word, pos, chunk, label) 二元组

special_prepare() 根据 template 采样实体进行填充、生成数据

() 表示可省去，[] 表示可替换同音、同义词，make_name() 组合人名

采样 general 的测试数据、添加到 special 的全体数据，减少过拟合

将 extra 添加到 special 的训练数据，train 70% / dev 20% / test 10% 划分

#### 2.explore

统计词汇、长度、实体的频率，条形图可视化，计算 slot_per_sent 指标

#### 3.represent

merge_vectorize() 合并 general 与 special 得到 embed_mat 与 label_ind

label2ind() 增设标签 N，vectorize() 分别截取或填充为定长序列

#### 4.build

general 分别使用双向 rnn、rnn_crf，special 载入 general 模型后再训练

#### 5.recognize

predict() 填充为定长序列、每句返回 (word, pred) 的二元组

#### 6.interface

merge() 将 BIO 标签组合为实体，response() 返回 json 字符串