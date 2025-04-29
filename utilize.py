import numpy as np
from tensorflow.keras.layers import Dense, Input, dot
from tensorflow.keras.models import Model


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split(',')
            row = []
            for k in line:
                row.append(float(k))
            md_data.append(row)
        md_data = np.array(md_data)
        return md_data


# get the features of each node by integrating three different network
def get_feature(M_M, M_D, D_D):
    H1 = np.hstack((M_M, M_D))
    H2 = np.hstack((M_D.transpose(), D_D))
    H = np.vstack((H1, H2))
    print('The shape of H', H.shape)
    return H


# find the miRNA-disease part
def find_mi_D(edges):
    m_d = []
    for i in range(edges.shape[0]):
        if edges[i, 0] < 577 and edges[i, 1] > 576:
            m_d.append(edges[i, :])
        elif edges[i, 0] > 576 and edges[i, 1] < 577:
            m_d.append(edges[i, :])
    return m_d


# def BuildModel(train_x, train_y):
#     from tensorflow.keras.layers import Input, Dense, Dropout
#     # This returns a tensor
#     l = len(train_x[1])
#     inputs = Input(shape=(l,))
#
#     # a layer instance is callable on a tensor, and returns a tensor
#     x = Dense(128, activation='relu')(inputs)
#     x = Dense(64, activation='relu')(x)
#     predictions = Dense(1, activation='sigmoid')(x)
#
#     # This creates a model that includes
#     # the Input layer and three Dense layers
#     model = Model(inputs=inputs, outputs=predictions)
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy', 'AUC'])
#     model.fit(train_x, train_y)  # starts training
#     return model

def BuildModel(train_x, train_y):
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2, l1
    # 输入层
    l = train_x.shape[1]
    inputs = Input(shape=(l,))
    # 隐藏层
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    # 输出层
    predictions = Dense(1, activation='sigmoid')(x)

    # 模型定义
    model = Model(inputs=inputs, outputs=predictions)

    # 编译模型，使用Adam优化器和自定义学习率
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])

    # 训练模型，包含验证集和回调函数
    model.fit(train_x, train_y)

    return model

# def BuildModel(train_x, train_y):
#     import lightgbm as lgb
#
#     # 创建 LightGBM 分类器
#     model = lgb.LGBMClassifier(
#         boosting_type='goss',
#         objective='binary',
#         learning_rate=0.01,
#         num_leaves=31,
#         n_estimators=1000,
#         verbosity=-1
#     )
#
#     # 训练模型
#     model.fit(train_x, train_y)
#
#     # 返回模型
#     return model
