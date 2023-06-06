import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt

        #recent模块，自定义模块GCGRU
        self.recent_network = GCGRUModel(opt)
        #长期模块 输入大小为opt['num_feature']
        self.trend_network = \
            nn.LSTM(input_size=opt['num_feature'], hidden_size=opt['hidden_dim'], num_layers=opt['num_layer'],
                    batch_first=True)
        # recent模块 全连接神经网层
        self.recent_fc = nn.Sequential(
            nn.Linear(in_features=opt['hidden_dim'] * opt['num_adj'] // 2, out_features=opt['hidden_dim']),
            nn.ReLU(),
        )
        # 全连接神经网络层 trend模块
        self.trend_fc = nn.Sequential(
            nn.Linear(in_features=opt['hidden_dim'], out_features=opt['hidden_dim']),
            nn.ReLU(),
        )
        #全连接层 外部特征模块
        self.feature_fc = nn.Sequential(
            nn.Linear(in_features=opt['time_feature'], out_features=opt['hidden_dim']),
            nn.ReLU(),
        )

        #
        #fusion层
        self.fc = nn.Sequential(
            nn.Linear(in_features=int(opt['hidden_dim'] * 2.5), out_features=opt['hidden_dim']),
            nn.ReLU(),
            nn.Linear(in_features=opt['hidden_dim'], out_features=opt['num_feature']),
            nn.Tanh()
        )
        #Gcn
        self.gcn = GCN(opt, input_size=int(opt['hidden_dim'] * 2.5), output_size=opt['num_feature'], activation='tanh')

    def forward(self, recent_data, trend_data, sub_graph, time_feature):
        """Generator
        :param recent_data: (B, seq_len, num_node, input_dim) #序列长度，点数量，输入维度(T,n,F)
        :param trend_data: (B, seq_len, input_dim)(T,F)
        :param sub_graph: (B, num_nodes, num_nodes)(n,n)
        :param time_feature: (B, time_features)
        :return
        - Output: `2-D` tensor with shape `(B, input_dim)`
        """

        batch_size = recent_data.shape[0]      #B是批处理大小
        #1
        recent, _ = self.recent_network(recent_data, sub_graph)  # (B, num_adj, rnn_units)
        #2
        trend, _ = self.trend_network(trend_data)  # (B, seq_len, hidden_dim)
        #提取最后一个时间步
        trend = trend[:, -1, ].view(batch_size, 1, -1)  # (B, hidden_dim) --> (B, 1, hidden_dim)
        #这样做是为了将每个样本的最后一个时间步的数据扩展到与 recent 数据相同的维度，以便后续的拼接操作。
        trend = trend.repeat(1, self.opt['num_adj'], 1)  # (B, num_adj, hidden_dim)
        #3
        feature_fc = self.feature_fc(time_feature).view(batch_size, 1, -1)  # (B, hidden_dim) --> (B, 1, hidden_dim)
        feature_fc = feature_fc.repeat(1, self.opt['num_adj'], 1)  # (B, num_adj, hidden_dim)
        #4
        #沿着第二个维度进行拼接
        combined = torch.cat([recent, trend, feature_fc], dim=2)
        #5
        output = self.gcn(combined, sub_graph)

        return output


class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        self.T_recent = self.opt['recent_time'] * self.opt['timestamp']
        #GCN
        self.gcn = GCN(opt, input_size=opt['num_feature'], output_size=opt['hidden_dim'])

        #GCGRU
        self.seq_network = GCGRUModel(opt)
        #FC
        self.seq_fc = nn.Sequential(
            nn.Linear(in_features=opt['hidden_dim'] * opt['num_adj'] // 2, out_features=opt['hidden_dim']),
            nn.ReLU(),
        )
        #？
        self.trend_network = \
            nn.LSTM(input_size=opt['num_feature'], hidden_size=opt['hidden_dim'], num_layers=opt['num_layer'],
                    batch_first=True)
        #fusion
        self.output = nn.Sequential(
            nn.Linear(in_features=opt['hidden_dim'] * 2, out_features=opt['hidden_dim']),
            nn.ReLU(),
            nn.Linear(in_features=opt['hidden_dim'], out_features=1),
            nn.Sigmoid()
        )

    def forward(self, sequence, sub_graph, trend_data):
        """Discrminator
        :param sequence: (B, seq_len, num_node, input_dim) or (seq_len, B, num_node, input_dim)
        :param sub_graph: (B, num_nodes, num_nodes)
        :param trend_data: (B, seq_len, input_dim)
        :return
        - Output: `2-D` tensor with shape `(B, 2)`
        """
        #GCGRU
        #因此，sequence[:, :-1, ] 会返回一个形状为 (B, seq_len-1, num_node, input_dim) 的张量
        seq, hid = self.seq_network(sequence[:, :-1, ], sub_graph)  # (B, num_adj, rnn_units)
        #将 seq 张量的形状从 (B, num_adj, rnn_units) 改变为 (B, num_adj * rnn_units)，然后通过 self.seq_fc 进行线性变换，得到 seq_fc 张量，形状为 (B, hidden_dim)，其中 hidden_dim 是全连接层的输出维度
        #FC
        #此处的-1代表自动计算
        seq_fc = self.seq_fc(seq.view(sequence.shape[0], -1))  # (B, hidden_dim)

        gcn = self.gcn(sequence[:, -1, ], sub_graph)  # (B, num_adj, hidden_dim)
        gcn_pooling = torch.max(gcn, dim=1)[0].squeeze()  # (B, hidden_dim)

        output = self.output(torch.cat([gcn_pooling, seq_fc], dim=1))
        return output


class GCN(nn.Module):
    def __init__(self, opt, input_size, output_size, activation='sigmoid'):
        super().__init__()
        self.opt = opt
        self.output_size = output_size

        self.fc = nn.Linear(in_features=input_size, out_features=output_size)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x, A):
        """Graph Convolution for batch.
        :param inputs: (B, num_nodes, input_dim)
        :param norm_adj: (B, num_nodes, num_nodes)
        :return
        - Output: `3-D` tensor with shape `(B, num_nodes, rnn_units)`
        """
        #x为输入数据
        #批次矩阵乘法 torch.bmm(A, x) 将输入数据 x 与邻接矩阵 A 进行乘法运算。这实际上是对输入数据 x 在图结构上进行卷积操作，根据邻接矩阵定义了节点之间的连接关系。
        x = torch.bmm(A, x)  # (B, num_nodes, input_dim)
        #乘法运算的结果输入到全连接层 self.fc 中进行线性变换。全连接层将输入的维度从 input_dim 映射到 output_size，
        x = self.fc(x)

        return self.activation(x)  # (B, num_nodes, rnn_units)


class GCGRUCell(torch.nn.Module):
    def __init__(self, opt, input_dim, rnn_units):
        super().__init__()

        input_size = input_dim + rnn_units

        self.r_gconv = GCN(opt, input_size=input_size, output_size=rnn_units)
        self.u_gconv = GCN(opt, input_size=input_size, output_size=rnn_units)
        self.c_gconv = GCN(opt, input_size=input_size, output_size=rnn_units, activation='tanh')

        def forward(self, x, h, A):
            """Gated recurrent unit (GRU) with Graph Convolution.
            :param inputs: (B, num_nodes, input_dim)
            :param hx: (B, num_nodes, rnn_units)
            :param norm_adj: (B, num_nodes, num_nodes)
            :return
            - Output: A `3-D` tensor with shape `(B, num_nodes, rnn_units)`.
            """
            #在某一维度上进行拼接  (B, num_nodes, input_dim + rnn_units)。
            x_h = torch.cat([x, h], dim=2)
            #计算重置门以及更新们 (B, num_nodes, rnn_units)
            r = self.r_gconv(x_h, A)
            u = self.u_gconv(x_h, A)
            #进行拼接 (B, num_nodes, input_dim + rnn_units)。 逐个元素相乘
            #看作是对过去状态 h 进行加权调节，其中权重由重置门 r 决定。乘积结果反映了过去状态的重要性以及保留或遗忘过去状态的程度。
            x_rh = torch.cat([x, r * h], dim=2)
            #图卷积操作 (B, num_nodes, rnn_units)
            c = self.c_gconv(x_rh, A)

            #1.0 - u) * c 表示对新状态 c 进行加权调节，其中权重由更新门的补数 1.0 - u 决定。乘积结果体现了新状态的重要性以及对过去状态的补充或替代程度。
            h = u * h + (1.0 - u) * c

            return h


class GCGRUModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_rnn_layers = opt['num_layer']

        self.num_nodes = opt['num_adj']
        self.rnn_units = opt['hidden_dim'] // 2

        self.dcgru_layers = nn.ModuleList(
            [GCGRUCell(opt=opt, input_dim=opt['num_feature'], rnn_units=self.rnn_units),
             GCGRUCell(opt=opt, input_dim=self.rnn_units, rnn_units=self.rnn_units)])

    def forward(self, inputs, norm_adj):
        """encoder forward pass on t time steps
        :param inputs: shape (batch_size, seq_len, num_node, input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        seq_len = inputs.shape[1]
        encoder_hidden_state = None
        for t in range(seq_len):
            output, encoder_hidden_state = self.encoder(inputs[:, t, ], norm_adj, encoder_hidden_state)

        return output, encoder_hidden_state

    def encoder(self, inputs, norm_adj, hidden_state=None):
        """Encoder
        :param inputs: shape (batch_size, self.num_nodes, self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.num_nodes, self.rnn_units)
               optional, zeros if not provided
        :return: output: `2-D` tensor with shape (B, self.num_nodes, self.rnn_units)
                 hidden_state: `2-D` tensor with shape (num_layers, B, self.num_nodes, self.rnn_units)
        """
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.num_nodes, self.rnn_units), device='cuda')
        hidden_states = []

        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num,], norm_adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)