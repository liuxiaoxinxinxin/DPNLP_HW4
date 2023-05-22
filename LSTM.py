# 下载语料

from tensorflow import keras
import numpy as np

path = keras.utils.get_file(
    'nietzsche.txt', 
    origin='')
text = open(path).read().lower()
print('Corpus length:', len(text))
# 将字符序列向量化

maxlen = 60     # 每个序列的长度
step = 3        # 每 3 个字符采样一个新序列
sentences = []  # 保存所提取的序列
next_chars = [] # sentences 的下一个字符

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i+maxlen])
    next_chars.append(text[i+maxlen])
print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))
char_indices = dict((char, chars.index(char)) for char in chars)
# 插：上面这两行代码 6
print('Unique characters:', len(chars))

print('Vectorization...')

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
# 用于预测下一个字符的单层 LSTM 模型

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
# 模型编译配置

from tensorflow.keras import optimizers

optimizer = optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)
def sample(preds, temperature=1.0):
    '''
    对模型得到的原始概率分布重新加权，并从中抽取一个字符索引
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
# 文本生成循环

import random

for epoch in range(1, 60):    # 训练 60 个轮次
    print(f'👉\033[1;35m epoch {epoch} \033[0m')    # print('epoch', epoch)
    
    model.fit(x, y,
              batch_size=128,
              epochs=1)
    
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print(f'  📖 Generating with seed: "\033[1;32;43m{generated_text}\033[0m"')    # print(f' Generating with seed: "{generated_text}"')
    
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print(f'\n   \033[1;36m 🌡️ temperature: {temperature}\033[0m')    # print('\n  temperature:', temperature)
        print(generated_text, end='')
        for i in range(400):    # 生成 400 个字符
            # one-hot 编码目前有的文本
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1
            
            # 预测，采样，生成下一字符
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            print(next_char, end='')
            
            generated_text = generated_text[1:] + next_char
            
    print('\n' + '-' * 20)
import random
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras
import numpy as np

import jieba    # 使用 jieba 做中文分词
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 导入文本

path = '~/Desktop/txt_zh_cn.txt'
text = open(path).read().lower()
print('Corpus length:', len(text))

# 将文本序列向量化

maxlen = 60     # 每个序列的长度
step = 3        # 每 3 个 token 采样一个新序列
sentences = []  # 保存所提取的序列
next_tokens = []  # sentences 的下一个 token

token_text = list(jieba.cut(text))

tokens = list(set(token_text))
tokens_indices = {token: tokens.index(token) for token in tokens}
print('Number of tokens:', len(tokens))

for i in range(0, len(token_text) - maxlen, step):
    sentences.append(
        list(map(lambda t: tokens_indices[t], token_text[i: i+maxlen])))
    next_tokens.append(tokens_indices[token_text[i+maxlen]])
print('Number of sequences:', len(sentences))

# 将目标 one-hot 编码
next_tokens_one_hot = []
for i in next_tokens:
    y = np.zeros((len(tokens),), dtype=np.bool)
    y[i] = 1
    next_tokens_one_hot.append(y)

# 做成数据集
dataset = tf.data.Dataset.from_tensor_slices((sentences, next_tokens_one_hot))
dataset = dataset.shuffle(buffer_size=4096)
dataset = dataset.batch(128)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# 构建、编译模型

model = models.Sequential([
    layers.Embedding(len(tokens), 256),
    layers.LSTM(256),
    layers.Dense(len(tokens), activation='softmax')
])

optimizer = optimizers.RMSprop(lr=0.1)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)

# 采样函数

def sample(preds, temperature=1.0):
    '''
    对模型得到的原始概率分布重新加权，并从中抽取一个 token 索引
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 训练模型

callbacks_list = [
    keras.callbacks.ModelCheckpoint(  # 在每轮完成后保存权重
        filepath='text_gen.h5',
        monitor='loss',
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(  # 不再改善时降低学习率
        monitor='loss',
        factor=0.5,
        patience=1,
    ),
    keras.callbacks.EarlyStopping(  # 不再改善时中断训练
        monitor='loss',
        patience=3,
    ),
]

model.fit(dataset, epochs=30, callbacks=callbacks_list)

# 文本生成

start_index = random.randint(0, len(text) - maxlen - 1)
generated_text = text[start_index: start_index + maxlen]
print(f' 📖 Generating with seed: "{generated_text}"')

for temperature in [0.2, 0.5, 1.0, 1.2]:
    print('\n  🌡️ temperature:', temperature)
    print(generated_text, end='')
    for i in range(100):    # 生成 100 个 token
        # 编码当前文本
        text_cut = jieba.cut(generated_text)
        sampled = []
        for i in text_cut:
            if i in tokens_indices:
                sampled.append(tokens_indices[i])
            else:
                sampled.append(0)

        # 预测，采样，生成下一个 token
        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_token = tokens[next_index]
        print(next_token, end='')

        generated_text = generated_text[1:] + next_token

