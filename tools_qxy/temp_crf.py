# ----- 系统操作相关 ----- #
import time
import os

# ----- 数据处理相关 ----- #
import numpy as np
import torch 
import torch.nn as nn

import torch.optim as optim 

'''在训练过程中需要用到的函数'''
def prepare_sequence(seq, to_ix):
    #seq是分词后语料，to_ix是语料库每个词对应的编号
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5 # 由于标签一共有B\I\O\START\STOP 5个，所以embedding_dim为5
HIDDEN_DIM = 4 # 这其实是BiLSTM的隐藏层的特征数量，因为是双向所以是2倍，单向为2

def argmax(vec):
    # 得到最大的值的索引
    _, idx = torch.max(vec, 1) # 返回每行中最大的元素和最大元素的索引
    return idx.item()


# 计算log部分的值  
def log_sum_exp(vec): #vec维度为1*5
    max_score = vec[0, argmax(vec)]#max_score的维度为1
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) #维度为1*5
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    #等同于torch.log(torch.sum(torch.exp(vec)))，防止e的指数导致计算机上溢

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        '''
        初始化模型
        parameters：
                vocab_size：语料的字典的长度
                tag_to_ix：标签与对应编号的字典
                embedding_dim：标签的数量
                hidden_dim：BiLSTM的隐藏层的神经元数量
        '''
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        #输出为一个mini-batch*words num*embedding_dim的矩阵
        #vocab_size表示一共有多少词，embedding_dim表示想为每个词创建一个多少维的向量来表示
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        #创建LSTM(每个词的特征数量，隐藏层的特征数量，循环层的数量指堆叠的层数，是否是BLSTM)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim//2,num_layers=1,bidirectional=True)

        # 将LSTM的输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转移矩阵，transitions[i][j]表示从label j转移到label i的概率,虽然是随机生成的但是后面会迭代更新
        # 这里特别要注意的是transitions[i]表示的是其他标签转到标签i的概率
        self.transitions = nn.Parameter(torch.randn(self.tagset_size,self.tagset_size))

        # 这两个语句强制执行了这样的约束条件：我们永远不会转移到开始标签，也永远不会从停止标签转移
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000#从任何标签转移到START_TAG不可能
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000#从STOP_TAG转移到任何标签不可能

        self.hidden = self.init_hidden()
                            
    def init_hidden(self):
        #（num_layers*num_directions,minibatch_size,hidden_dim）
        # 实际上初始化的h0和c0
        return (torch.randn(2, 1, self.hidden_dim // 2),
        
                torch.randn(2, 1, self.hidden_dim // 2))
    def forward(self, sentence):  #sentence是已经编码的句子
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)    
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
    def _viterbi_decode(self, feats):
        #预测序列的得分，维特比解码，输出得分与路径值
        backpointers = []

        # Initialize the viterbi variables
        init_vvars = torch.full((1, self.tagset_size), -10000.)#这就保证了一定是从START到其他标签
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                next_tag_var = forward_var + self.transitions[next_tag]#forward_var保存的是之前的最优路径的值
                best_tag_id = argmax(next_tag_var) #返回最大值对应的那个tag
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 从step0到step(i-1)时5个序列中每个序列的最大score
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)# bptrs_t有５个元素

        # 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()# 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):# loss function
        feats = self._get_lstm_features(sentence) # 经过LSTM+Linear后的输出作为CRF的输入
        forward_score = self._forward_alg(feats) # loss的log部分的结果
        gold_score = self._score_sentence(feats, tags)# loss的后半部分S(X,y)的结果
        print("feats: ", feats.shape)
        print("forward_score: ", forward_score.shape) 
        print("gold_score: ", gold_score.shape)

        print("forward_score - gold_score", forward_score - gold_score)
        return forward_score - gold_score #Loss
    def _get_lstm_features(self, sentence):#仅仅是BiLSTM的输出没有CRF层
        self.hidden = self.init_hidden()  # 一开始的隐藏状态
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) #最后输出结果和最后的隐藏状态
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    def _forward_alg(self, feats):#预测序列的得分，就是Loss的右边第一项
        #feats表示发射矩阵(emit score)，实际上就是LSTM的输出，意思是经过LSTM的sentence的每个word对应于每个label的得分
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.) #用-10000.来填充一个形状为[1,tagset_size]的tensor
            
        # START_TAG has all of the score.
        # 因为start tag是4，所以tensor([[-10000., -10000., -10000., 0., -10000.]])，
        # 将start的值为零，表示开始进行网络的传播，
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 包装到一个变量里面以便自动反向传播
        forward_var = init_alphas  # 初始状态的forward_var，随着step t变化

        # 遍历句子，迭代feats的行数次
        for feat in feats:
            alphas_t = []  # 当前时间步的正向tensor
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                #LSTM的生成矩阵是emit_score，维度为1*5
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the i_th entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)#维度是1*5
                # 第一次迭代时理解：
                # trans_score是所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is logsumexp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                # 此时的alphas t 是一个长度为5，例如<class 'list'>:
                # [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
            forward_var = torch.cat(alphas_t).view(1, -1)
        # 最后只将最后一个单词的forward var与转移 stop tag的概率相加
        # tensor([[   21.1036,    18.8673,    20.7906, -9982.2734, -9980.3135]])
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)# alpha是一个0维的tensor
        return alpha
    def _score_sentence(self, feats, tags):# 求Loss function的第二项
        # 这与上面的def _forward_alg(self, feats)共同之处在于：两者都是用的随机转移矩阵算的score，不同地方在于，上面那个函数算了一个最大可能路径，但实际上可能不是真实的各个标签转移的值 例如：真实标签是N V V,但是因为transitions是随机的，所以上面的函数得到其实是N N N这样，两者之间的score就有了差距。而后来的反向传播，就能够更新transitions，使得转移矩阵逼近真实的“转移矩阵”得到gold_seq tag的score 即根据真实的label 来计算一个score，但是因为转移矩阵是随机生成的，故算出来的score不是最理想的值
        score = torch.zeros(1)
        # 将START_TAG的标签３拼接到tag序列最前面
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，
            # 对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            # transition【j,i】 就是从i ->j 的转移概率值
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score



# 训练数据
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(), #['the','wall','street','journal',...]
    "B I I I O O O B I O O".split() #['B','I','I',...]
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

# 给每一个不重复的词进行编码，比如‘HELLO WORLD’就是{'HELLO':0,'WORLD:1'}
word_to_ix = {} # 训练语料的字典，语料中的每一个字对应的编码(index)
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}# tag的字典，每个tag对应一个编码

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM) #模型
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4) #优化器，采用随机梯度下降

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    # print(model(precheck_sent)) #(tensor(2.6907), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1])

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(1): #循环次数可以自己设定
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # print(sentence_in)
        # print(targets)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!

