"""
    本代码用于: 用于定义 LSTM-CRF 模型
    创建时间: 2021 年 11 月 26 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 11 月 26 日
"""

# ==================== 导入必要的包 ==================== #
# ----- 模型创建相关的 ----- #
import torch
import torch.nn.functional as F  
import torch.nn as nn

# ----- 定义常量 ----- # 
START_TAG = "<START>"
STOP_TAG = "<STOP>"

# ==================== 函数实现 ==================== #
# ----- 得到最大的值的索引 ----- #
def argmax(vec):
    _, idx = torch.max(vec, 1) # 返回每行中最大的元素和最大元素的索引
    return idx.item()

# ----- 计算 log 部分的值 ----- #
def log_sum_exp(vec): #vec维度为1*5
    max_score = vec[0, argmax(vec)]#max_score的维度为1
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) #维度为1*5
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# ----- 定义模型 ----- #
class NERLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NERLSTM, self).__init__()

        # ----- 初始化参数 ----- # 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id)
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)

        # ----- 建立词向量表 ----- #
        # self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeds = nn.Linear(300, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # ----- 建立 LSTM 模型 ----- #
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        # ----- 将 LSTM 的输出映射到标签空间 ----- #
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # ----- 转移矩阵 ----- #
        self.transitions = nn.Parameter(torch.randn(self.tagset_size,self.tagset_size))
        # 从 STOP_TAG 转移到任何标签不可能 && 从任何标签转移到 START_TAG 不可能
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

        # ----- 初始化 ----- #
        self.hidden = self.init_hidden()

    # ----- 前馈过程 ----- #
    def forward(self, x):  
        lstm_feats = self._get_lstm_features(x) # 经过LSTM+Linear后的输出作为CRF的输入

        tag_seq = torch.tensor(self._viterbi_decode(lstm_feats[0])).view(1,lstm_feats.shape[1])
        for i in range(1, lstm_feats.shape[0]):
            q=torch.tensor(self._viterbi_decode(lstm_feats[i])).view(1,lstm_feats.shape[1])
            tag_seq = torch.cat((tag_seq, q),dim = 0 )
        return tag_seq


    # ----- 初始化的 h0 和 c0 ----- #
    def init_hidden(self):
        #（num_layers*num_directions,minibatch_size,hidden_dim）
        return (torch.randn(2, 1, self.hidden_dim // 2),        
                torch.randn(2, 1, self.hidden_dim // 2))


    # ----- 仅仅是 BiLSTM 的输出没有 CRF 层 ----- #
    def _get_lstm_features(self, sentence):
        embeds = self.word_embeds(sentence)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    # ----- 隐藏层初始化 ----- #
    def init_hidden(self):
        #（num_layers * num_directions, minibatch_size, hidden_dim）
        # 实际上初始化的 h0 和 c0
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _viterbi_decode(self, feats):
        
        # 预测序列的得分，维特比解码，输出得分与路径值
        backpointers = []

        # Initialize the viterbi variables
        init_vvars = torch.full((1, self.tagset_size), -10000.)#这就保证了一定是从START到其他标签
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        forward_var = forward_var.cuda()
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                # print(forward_var) 
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

        return best_path


    # ----- 计算 loss Function ----- #
    def neg_log_likelihood(self, sentence, tags):# loss function
        feats = self._get_lstm_features(sentence) # 经过LSTM+Linear后的输出作为CRF的输入
        print("feats: ", feats.shape)
        
        forward_score = self._forward_alg(feats) # loss的log部分的结果
        print("forward_score: ", forward_score.shape) 
        
        gold_score = self._score_sentence(feats, tags)# loss的后半部分S(X,y)的结果
        print("gold_score: ", gold_score.shape)
        

        return forward_score - gold_score #Loss


    # ----- 预测序列的得分，就是 Loss 的右边第一项 ----- #
    def _forward_alg(self, feats):
        #feats表示发射矩阵(emit score)，实际上就是LSTM的输出，意思是经过LSTM的sentence的每个word对应于每个label的得分
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.) #用-10000.来填充一个形状为[1,tagset_size]的tensor
            
        # START_TAG has all of the score.
        # 因为start tag是4，所以tensor([[-10000., -10000., -10000., 0., -10000.]])，
        # 将start的值为零，表示开始进行网络的传播，
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 包装到一个变量里面以便自动反向传播
        forward_var = init_alphas  # 初始状态的forward_var，随着step t变化
        forward_var = forward_var.cuda()

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

    # ----- 求 Loss function 的第二项 ----- #
    def _score_sentence(self, feats, tags):
        # 这与上面的def _forward_alg(self, feats)共同之处在于：两者都是用的随机转移矩阵算的score，不同地方在于，上面那个函数算了一个最大可能路径，但实际上可能不是真实的各个标签转移的值 例如：真实标签是N V V,但是因为transitions是随机的，所以上面的函数得到其实是N N N这样，两者之间的score就有了差距。而后来的反向传播，就能够更新transitions，使得转移矩阵逼近真实的“转移矩阵”得到gold_seq tag的score 即根据真实的label 来计算一个score，但是因为转移矩阵是随机生成的，故算出来的score不是最理想的值
            score = torch.zeros(1)
            score = score.cuda()

            # 将START_TAG的标签３拼接到tag序列最前面
            tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags.cpu()])
            tags = tags.cuda()
            # print(self.transitions.device)

            for i, feat in enumerate(feats):
                # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
                # feat[tags[i+1]], feat是step i 的输出结果，有５个值，
                # 对应B, I, E, START_TAG, END_TAG, 取对应标签的值
                # transition【j,i】 就是从i ->j 的转移概率值
                score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
            return score

    def neg_log_likelihood_tensor(self, sentence, tags):  # loss function
        feats = self._get_lstm_features(sentence) # 经过LSTM+Linear后的输出作为CRF的输入
        len_feats = feats.shape[0]

        forward_score = self._tensor_forward_alg(feats)
        gold_score = self._tensor_score_sentence(feats, tags)

        q = torch.sum(forward_score - gold_score)
        ans = q / len_feats

        return ans

    def _tensor_score_sentence(self, feats, tags):  # 求Loss function的第二项

        score = torch.zeros(feats.shape[0]).cuda()
        score = score.view(feats.shape[0], 1)
        star_tag = torch.tensor([self.tag_to_ix[START_TAG]], dtype = torch.long).expand(feats.shape[0],
                                                                                        1).cuda()  # shape=1
        # tags=32,100
        tags = torch.cat([star_tag, tags], 1)
        star_tag = torch.tensor([self.tag_to_ix[STOP_TAG]], dtype = torch.long).expand(feats.shape[0], 1).cuda()
        tags = torch.cat([tags, star_tag], 1)
        # transitions = self.transitions.expand(32, self.tagset_size, self.tagset_size)
        len_s = feats.shape[1] + 1  # 获取长度
        check = torch.arange(feats.shape[0])
        # 遍历句子，迭代feats的行数次
        for i in range(len_s):
            tag_now = tags[:, i]  # tag=[32]
            tag_nex = tags[:, i + 1]
            # score = score + transitions[tags[i + 1], tags[i]].expand(32, 1) + feat[:,tags[i + 1]]  # feat里没行的tags[i+1]列信息
            temp = torch.index_select(self.transitions, 0, tag_nex)  # temp.shape=[32,10] ,第i行就是第i个batch的转移数组
            temp = temp[check, tag_now].view(feats.shape[0], 1)
            score += temp
            if i <= len_s - 2:
                feat = feats[:, i, :]  # feat=32*10
                score += feat[check, tag_nex].view(feats.shape[0], 1)
            # score = score + + feat[:,tags[i + 1]]  # feat里没行的tags[i+1]列信息

        return score

    def _tensor_forward_alg(self, feats):  # 预测序列的得分，就是Loss的右边第一项#all cuda
        init_alphas = torch.full((feats.shape[0], self.tagset_size), -10000.).cuda()
        # 用-10000.来填充一个形状为[batch,tagset_size]的tensor
        # 若batch=2, tags.len是5 star_tag=4，则tensor([[-10000., -10000., -10000., 0., -10000.],
        #                                           [-10000., -10000., -10000., 0., -10000.] ])，
        # 将start的值为零，表示开始进行网络的传播，
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas  # 初始状态的forward_var，随着step t变化
        len_s = feats.shape[1]  # 获取长度
        # 遍历句子，迭代feats的行数次
        for i in range(len_s):
            feat = feats[:, i, :]
            # 获取第一个标签下的loss
            emit_score = feat[:, 0].view(feats.shape[0], 1).expand(feats.shape[0], self.tagset_size)  # cuda
            trans_score = self.transitions[0].view(1, -1)  # 维度是1*10,#cuda
            trans_score = trans_score.expand(feats.shape[0], self.tagset_size)  # 维度是32*10,#cuda
            next_tag_var = forward_var + trans_score + emit_score  # cuda
            temp = self._tensor_log_sum_exp(next_tag_var).view(1, feats.shape[0])

            for next_tag in range(1, self.tagset_size):
                # feat[当前标签]的扩展矩阵是emit_score，维度为32*10
                # e=e.expand(2,a.shape[2])
                emit_score = feat[:, next_tag].view(feats.shape[0], 1).expand(feats.shape[0], self.tagset_size)  # cuda
                trans_score = self.transitions[next_tag].view(1, -1)  # 维度是1*10,#cuda
                trans_score = trans_score.expand(feats.shape[0], self.tagset_size)  # 维度是32*10,#cuda
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊10，10个值是相同的
                next_tag_var = forward_var + trans_score + emit_score  # cuda

                # 需要.view()是因为返回值本身是0维的，然后将他们cat在一起
                temp = torch.cat((temp, self._tensor_log_sum_exp(next_tag_var).view(1, feats.shape[0])), dim = 0)



            forward_var = temp.transpose(0, 1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]  # cuda
        alpha = self._tensor_log_sum_exp(terminal_var).view(feats.shape[0], 1)  # alpha是一个0维的tensor，这里变成【32,1】维度的格式

        return alpha  # cuda

    def _tensor_log_sum_exp(self, vec):  # vec维度为batch*10
        # max_score = vec[0, self.argmax(vec)]  # max_score的维度为1

        max_score, idx = torch.max(vec, 1)  # max_ . shape=[1,batch]
        max_score_broadcast = max_score.expand(vec.shape[1], vec.shape[0])  # 维度为10*batch
        # torch.exp将所有元素求exp
        # torch.sum是求tensor的所有元素和
        # torch.log对所有元素求log
        vec = vec.transpose(0, 1)

        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim = 0))  # 每一列相加
