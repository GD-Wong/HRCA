import math
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

##############################
##  残差块儿ResidualBlock
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features, batchnorm = True):
        super(ResidualBlock, self).__init__()
        # Res层标准化方法
        if batchnorm:
            res_norm_layer = nn.BatchNorm1d
        else:
            res_norm_layer = nn.LayerNorm
        self.block = nn.Sequential(  ## block = [Linear  + norm + relu + Linear + norm]
            nn.Linear(in_features, in_features),
            res_norm_layer(in_features),
            nn.ReLU(inplace=True),  ## 非线性激活
            nn.Linear(in_features, in_features),
            res_norm_layer(in_features),
        )

    def forward(self, x):  ## 输入为 一张图像
        return x+self.block(x)                     ##累加输出
        #return torch.cat([self.block(x), x], dim=1)  ## 连接输出


##############################
##  生成器网络GeneratorResNet
##############################
class GeneratorResNet(nn.Module):
    def __init__(self, input_features, nres, batchnorm = True, init_features = 5000):  ## input_features = feature_size
        super(GeneratorResNet, self).__init__()
        # 生成器标准化层
        if batchnorm:
            gen_norm_layer = nn.BatchNorm1d
        else:
            gen_norm_layer = nn.LayerNorm
        model = [
            nn.Linear(input_features, init_features),
            gen_norm_layer(init_features),
            nn.ReLU(inplace=True),  ## 非线性激活
        ]
        out_features = init_features

        ## 编码
        for _ in range(2):
            model += [  ## (Conv + Norm + ReLU) * 2
                nn.Linear(out_features, out_features//2),
                gen_norm_layer(out_features//2),
                nn.ReLU(inplace=True),
            ]
            out_features//=2

        # 残差块儿，循环n_res次,残差层外套ReLU激活
        for _ in range(nres):
            model += [ResidualBlock(out_features), nn.Linear(out_features * 2, out_features), nn.ReLU(inplace=True)]

        # 解码
        for _ in range(2):
            model += [
                nn.Linear(out_features, out_features*2),
                gen_norm_layer(out_features*2),
                nn.ReLU(inplace=True),
            ]
            out_features*=2

        ## 网络输出层                                                            ## model += [pad + conv + tanh]
        model += [nn.Linear(out_features, input_features), nn.Softplus()]
        self.model = nn.Sequential(*model)

    def forward(self, x):  ## 输入(batch_size, feature_size)
        return self.model(x)  ## 输出(batch_size, feature_size)


##############################
##  单层编码
##############################
class EncoderBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, activation=True, norm_flag=True, norm_layer = nn.LayerNorm):
        super(EncoderBlock, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.norm_flag = norm_flag
        self.nl = norm_layer(output_size)

    def forward(self, x):
        if self.activation:
            out = self.linear(self.lrelu(x))
        else:
            out = self.linear(x)
        if self.norm_flag:
            return self.nl(out)
        else:
            return out

##############################
##  单层解码
##############################
class DecoderBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, norm_flag=True, dropout=False, norm_layer = nn.LayerNorm):
        super(DecoderBlock, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.nl = norm_layer(output_size)
        self.relu = torch.nn.ReLU(True)
        self.norm_flag = norm_flag
        self.dropout = dropout

    def forward(self, x):
        if self.norm_flag:
            out = self.nl(self.linear(self.relu(x)))
        else:
            out = self.linear(self.relu(x))
        if self.dropout:
            return F.dropout(out, 0.5, training=self.training)
        else:
            return out

##############################
##  UNET网络生成器
##############################
class GeneratorUNET(nn.Module):
    def __init__(self, input_shape, num_filter=5000, dropout_input = False, dropout_rate = 0.5):
        super(GeneratorUNET, self).__init__()
        self.dropout_input = dropout_input
        self.dropout_rate = dropout_rate
        # Input
        self.en1 = EncoderBlock(input_shape, num_filter, activation=False, norm_flag=False)
        # Encoder
        self.en2 = EncoderBlock(num_filter, num_filter // 2)
        self.en3 = EncoderBlock(num_filter // 2, num_filter // 4)
        self.en4 = EncoderBlock(num_filter // 4, num_filter // 8)

        self.en5 = EncoderBlock(num_filter // 8, num_filter // 8)
        self.en6 = EncoderBlock(num_filter // 8, num_filter // 8)
        self.en7 = EncoderBlock(num_filter // 8, num_filter // 8)
        self.en8 = EncoderBlock(num_filter // 8, num_filter // 8)
        # Decoder
        self.de1 = DecoderBlock(num_filter // 8, num_filter // 8, dropout=False)
        self.de2 = DecoderBlock(num_filter // 4, num_filter // 8, dropout=False)
        self.de3 = DecoderBlock(num_filter // 4, num_filter // 8, dropout=False)
        self.de4 = DecoderBlock(num_filter // 4, num_filter // 8)

        self.de5 = DecoderBlock(num_filter // 4, num_filter // 4)
        self.de6 = DecoderBlock(num_filter // 2, num_filter // 2)
        self.de7 = DecoderBlock(num_filter, num_filter)
        # Output
        self.de8 = DecoderBlock(num_filter * 2, input_shape, norm_flag=False)
        # activate
        self.activate = nn.Softplus()

    def forward(self, x):
        # Input
        if self.dropout_input:
            x = F.dropout(x, self.dropout_rate, training=self.training)
        enc1 = self.en1(x)
        # Encoder
        enc2 = self.en2(enc1)
        enc3 = self.en3(enc2)
        enc4 = self.en4(enc3)
        enc5 = self.en5(enc4)
        enc6 = self.en6(enc5)
        enc7 = self.en7(enc6)
        enc8 = self.en8(enc7)
        # Decoder with skip-connections
        dec = self.de1(enc8)
        dec = torch.cat([dec, enc7], 1)
        dec = self.de2(dec)
        dec = torch.cat([dec, enc6], 1)
        dec = self.de3(dec)
        dec = torch.cat([dec, enc5], 1)
        dec = self.de4(dec)
        dec = torch.cat([dec, enc4], 1)
        dec = self.de5(dec)
        dec = torch.cat([dec, enc3], 1)
        dec = self.de6(dec)
        dec = torch.cat([dec, enc2], 1)
        dec = self.de7(dec)
        dec = torch.cat([dec, enc1], 1)
        # Ountput with skip-connections
        dec = self.de8(dec)
        dec = self.activate(dec)
        return dec

class GeneratorMiniUNET(nn.Module):
    def __init__(self, input_shape, num_filter=5000, dropout_input = False, dropout_rate = 0.5):
        super(GeneratorMiniUNET, self).__init__()
        self.dropout_input = dropout_input
        self.dropout_rate = dropout_rate
        # Input
        self.en1 = EncoderBlock(input_shape, num_filter, activation=False, norm_flag=False)
        # Encoder
        self.en2 = EncoderBlock(num_filter, num_filter // 2)
        self.en3 = EncoderBlock(num_filter // 2, num_filter // 4)
        self.en4 = EncoderBlock(num_filter // 4, num_filter // 8)

        self.en5 = EncoderBlock(num_filter // 8, num_filter // 8)

        self.de1 = DecoderBlock(num_filter // 8, num_filter // 8, dropout=False)

        self.de2 = DecoderBlock(num_filter // 4, num_filter // 4)
        self.de3 = DecoderBlock(num_filter // 2, num_filter // 2)
        self.de4 = DecoderBlock(num_filter, num_filter)
        # Output
        self.de5 = DecoderBlock(num_filter * 2, input_shape, norm_flag=False)
        # activate
        self.activate = nn.Softplus()

    def forward(self, x):
        # Input
        if self.dropout_input:
            x = F.dropout(x, self.dropout_rate, training=self.training)
        enc1 = self.en1(x)
        # Encoder
        enc2 = self.en2(enc1)
        enc3 = self.en3(enc2)
        enc4 = self.en4(enc3)
        enc5 = self.en5(enc4)
        # Decoder with skip-connections
        dec = self.de1(enc5)
        dec = torch.cat([dec, enc4], 1)
        dec = self.de2(dec)
        dec = torch.cat([dec, enc3], 1)
        dec = self.de3(dec)
        dec = torch.cat([dec, enc2], 1)
        dec = self.de4(dec)
        dec = torch.cat([dec, enc1], 1)
        # Ountput with skip-connections
        dec = self.de5(dec)
        dec = self.activate(dec)
        return dec

class GeneratorUNETPlus(nn.Module):
    def __init__(self, input_shape, num_filter=5000, dropout_input = False, dropout_rate = 0.5):
        super(GeneratorUNETPlus, self).__init__()
        self.dropout_input = dropout_input
        self.dropout_rate = dropout_rate
        # Input
        self.en1 = EncoderBlock(input_shape, num_filter, activation=False, norm_flag=False)
        # Encoder
        self.en2 = EncoderBlock(num_filter, num_filter // 2)
        self.en3 = EncoderBlock(num_filter // 2, num_filter // 4)
        self.en4 = EncoderBlock(num_filter // 4, num_filter // 8)

        self.en5 = EncoderBlock(num_filter // 8, num_filter // 8)
        self.en6 = EncoderBlock(num_filter // 8, num_filter // 8)
        self.en7 = EncoderBlock(num_filter // 8, num_filter // 8)
        self.en8 = EncoderBlock(num_filter // 8, num_filter // 8)
        # Decoder
        self.de1 = DecoderBlock(num_filter // 8, num_filter // 8, dropout=False)
        self.de2 = DecoderBlock(num_filter // 8, num_filter // 8, dropout=False)
        self.de3 = DecoderBlock(num_filter // 8, num_filter // 8, dropout=False)
        self.de4 = DecoderBlock(num_filter // 8, num_filter // 8)

        self.de5 = DecoderBlock(num_filter // 8, num_filter // 4)
        self.de6 = DecoderBlock(num_filter // 4, num_filter // 2)
        self.de7 = DecoderBlock(num_filter // 2, num_filter)
        # Output
        self.de8 = DecoderBlock(num_filter, input_shape, norm_flag=False)
        # activate
        self.activate = nn.Softplus()

    def forward(self, x):
        # Input
        if self.dropout_input:
            x = F.dropout(x, self.dropout_rate, training=self.training)
        enc1 = self.en1(x)
        # Encoder
        enc2 = self.en2(enc1)
        enc3 = self.en3(enc2)
        enc4 = self.en4(enc3)
        enc5 = self.en5(enc4)
        enc6 = self.en6(enc5)
        enc7 = self.en7(enc6)
        enc8 = self.en8(enc7)
        # Decoder with skip-connections
        dec = self.de1(enc8)
        dec += enc7
        dec = self.de2(dec)
        dec += enc6
        dec = self.de3(dec)
        dec += enc5
        dec = self.de4(dec)
        dec += enc4
        dec = self.de5(dec)
        dec += enc3
        dec = self.de6(dec)
        dec += enc2
        dec = self.de7(dec)
        dec += enc1
        # Ountput with skip-connections
        dec = self.de8(dec)
        dec = self.activate(dec)
        return dec

##############################
##  UNET去除连接后的自编码器结构
##############################
class GeneratorAE(nn.Module):
    def __init__(self, input_shape, num_filter=5000, dropout_input=False, dropout_rate=0.5):
        super(GeneratorAE, self).__init__()
        self.dropout_input = dropout_input
        self.dropout_rate = dropout_rate
        # Input
        model = [EncoderBlock(input_shape, num_filter, activation=False, norm_flag=False),
                 EncoderBlock(num_filter, num_filter // 2),
                 EncoderBlock(num_filter // 2, num_filter // 4),
                 EncoderBlock(num_filter // 4, num_filter // 8),
                 EncoderBlock(num_filter // 8, num_filter // 8),
                 EncoderBlock(num_filter // 8, num_filter // 8),
                 EncoderBlock(num_filter // 8, num_filter // 8),
                 EncoderBlock(num_filter // 8, num_filter // 8),
                 DecoderBlock(num_filter // 8, num_filter // 8, dropout=False),
                 DecoderBlock(num_filter // 8, num_filter // 8, dropout=False),
                 DecoderBlock(num_filter // 8, num_filter // 8, dropout=False),
                 DecoderBlock(num_filter // 8, num_filter // 8),
                 DecoderBlock(num_filter // 8, num_filter // 4),
                 DecoderBlock(num_filter // 4, num_filter // 2),
                 DecoderBlock(num_filter // 2, num_filter),
                 DecoderBlock(num_filter, input_shape, norm_flag=False),
                 nn.Softplus()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        # Input
        if self.dropout_input:
            x = F.dropout(x, self.dropout_rate, training=self.training)
        return self.model(x)

class GeneratorMiniAE(nn.Module):
    def __init__(self, input_shape, num_filter=5000, dropout_input=False, dropout_rate=0.5):
        super(GeneratorMiniAE, self).__init__()
        self.dropout_input = dropout_input
        self.dropout_rate = dropout_rate
        # Input
        model = [EncoderBlock(input_shape, num_filter, activation=False, norm_flag=False),
                 EncoderBlock(num_filter, num_filter // 2),
                 EncoderBlock(num_filter // 2, num_filter // 4),
                 EncoderBlock(num_filter // 4, num_filter // 8),
                 DecoderBlock(num_filter // 8, num_filter // 4),
                 DecoderBlock(num_filter // 4, num_filter // 2),
                 DecoderBlock(num_filter // 2, num_filter),
                 DecoderBlock(num_filter, input_shape, norm_flag=False),
                 nn.Softplus()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        # Input
        if self.dropout_input:
            x = F.dropout(x, self.dropout_rate, training=self.training)
        return self.model(x)

##############################
##  带残差层的自编码器结构
##############################
class GeneratorResAE(nn.Module):
    def __init__(self, input_shape, nres = 4, num_filter=5000):
        super(GeneratorResAE, self).__init__()
        res_layer = []
        encoder = [EncoderBlock(input_shape, num_filter, activation=False, norm_flag=False),
                 EncoderBlock(num_filter, num_filter // 4),
                 EncoderBlock(num_filter // 4, num_filter // 8),
                 nn.LeakyReLU(0.2, True)]
        for _ in range(nres):
            res_layer +=  [ResidualBlock(num_filter // 8, batchnorm=False)]
        decoder = [DecoderBlock(num_filter // 8, num_filter // 4),
                  DecoderBlock(num_filter // 4, num_filter),
                  DecoderBlock(num_filter, input_shape, norm_flag=False),
                  nn.Softplus()]
        self.encoder = nn.Sequential(*encoder)
        self.res_layer = nn.Sequential(*res_layer)
        self.decoder = nn.Sequential(*decoder)
    def forward(self, x):
        emb = self.encoder(x)
        emb = self.res_layer(emb)
        return self.decoder(emb)
    def encode(self, x):
        return self.res_layer(self.encoder(x))


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.output_shape = 256
        def discriminator_block(in_filters, out_filters, normalize=True):           ## 鉴别器块儿
            """Returns downsampling layers of each discriminator block"""
            if normalize:                                                           ## 每次卷积尺寸会缩小一半，共卷积了4次
                layers = [nn.utils.spectral_norm(nn.Linear(in_filters, out_filters))]
            else:
                layers = [nn.Linear(in_filters, out_filters)]   ## layer += [conv + norm + relu]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_shape, 1024, normalize=False),
            *discriminator_block(1024, 512),
            *discriminator_block(512, 256),
            nn.utils.spectral_norm(nn.Linear(256,256))
        )

    def forward(self, img):             ## 输入(batch_size, feature_size)
        return self.model(img)          ## 输出(batch_size, 256)

##############################
#        SE Module
##############################
class SELayer(nn.Module):
    def __init__(self, input_shape, embedding_shape):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, embedding_shape, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_shape, input_shape, bias = False),
            nn.Sigmoid()
            #nn.Softmax()
        )
    def forward(self, x):
        y = self.fc(x)
        return y*x

class QSSELayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(QSSELayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)
    def forward(self, inputs):
        """
        inputs: batch_size x seq_len x input_size
        """
        #energy = torch.tanh(self.W1(inputs))
        energy = F.relu(self.W1(inputs))
        attention_weights = F.sigmoid(self.W2(energy))#, dim=1
        context_vectors = attention_weights * inputs
        #context_vectors = torch.sum(context_vectors, dim=1)
        return context_vectors

class QSNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(QSNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ff = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, input_size//8),
        )
        #self.attention = QSSELayer(input_size//8, input_size//16)
        self.attention = SELayer(input_size//8, input_size//16)
        self.fc1 = nn.Linear(input_size//8, input_size//16)
        self.fc2 = nn.Linear(input_size//16, output_size)
        #self.autoencoder = Autoencoder(input_size)
    def forward(self, inputs):
        x = self.ff(inputs)
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y


class SEResBlock(nn.Module):
    def __init__(self, input_shape, output_shape, dropout_rate = 0.0):
        super(SEResBlock, self).__init__()
        self.layer = utils.clones(nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_shape, output_shape),
            nn.BatchNorm1d(output_shape),
            #nn.LayerNorm(output_shape),
            nn.ReLU(inplace=True),
        ), 2)
        self.atten = SELayer(output_shape, output_shape//2)
    def forward(self, x):
        return self.layer[0](x)+self.atten(self.layer[1](x))

class SEFFNet(nn.Module):
    def __init__(self, input_shape, output_shape, dropout_rate = 0.0):
        super(SEFFNet, self).__init__()
        self.attn = SELayer(input_shape, input_shape // 2)
        #self.normLayer = nn.BatchNorm1d
        self.normLayer = nn.LayerNorm
        self.ff = nn.Sequential(
            nn.Dropout(p = dropout_rate),
            nn.Linear(input_shape, input_shape//4),
            self.normLayer(input_shape//4),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout_rate),
            nn.Linear(input_shape//4, input_shape//16),
            self.normLayer(input_shape // 16),
            nn.ReLU(inplace=True),
            nn.Linear(input_shape//16, output_shape)
        )
    def forward(self, x):
        out = self.attn(x)
        out = self.ff(out)
        return out

class SENet(nn.Module):
    def __init__(self, input_shape, output_shape, n_layers = 3):
        super(SENet, self).__init__()
        SERes_Layer = [SEResBlock(input_shape, input_shape//2, dropout_rate=0.0)]
        self.attn = SELayer(input_shape, input_shape//2)
        emb_shape = input_shape//2
        for _ in range(n_layers):
            SERes_Layer += [SEResBlock(emb_shape, emb_shape//2, dropout_rate=0.5)]
            emb_shape = emb_shape//2
        SERes_Layer += [nn.Linear(emb_shape, output_shape)]
        self.SEres = nn.Sequential(*SERes_Layer)
    def forward(self, x):
        out = self.attn(x)
        out = self.SEres(out)
        return out





##############################
#        Multi Head
##############################
class MultiHeadSENet(nn.Module):
    def __init__(self, input_shape, emb_shape, head_num = 8):
        super(MultiHeadSENet, self).__init__()
        assert emb_shape%head_num==0, "emb_shape can't split by head_num"
        self.head_num = head_num
        self.input_shape = input_shape
        self.emb_shape = emb_shape
        self.dk = emb_shape//head_num
        self.split_layer = nn.Linear(input_shape, emb_shape)
        self.attn_layer = SELayer(input_shape=head_num, embedding_shape=head_num//2)
        self.concat_layer = nn.Linear(emb_shape, input_shape)
    def forward(self, x):
        b = x.size(0)
        # [batch, input_shape] -> [batch, emb_shape(dk*head_num)]
        x = self.split_layer(x)
        # [batch, dk*head_num] -> [batch, dk, head_num]
        x = x.view(b, self.dk, self.head_num)
        x = self.attn_layer(x)
        # [batch, input_shape, head_num] -> [batch, input_shape*head_num]
        x = x.view(b, self.emb_shape)
        # [batch, head_num*input_shape] -> [batch, input_shape]
        y = self.concat_layer(x)
        return y

##############################
#        Attention Model
##############################
def attention(q, k, v, train_status, dropout = 0, mask=None):
    dk = q.size(-1)
    # add dim for matmul
    q = q.unsqueeze(dim = -1) #(batch, input_shape, 1)
    k = k.unsqueeze(dim = -1)
    v = v.unsqueeze(dim = -1)
    scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(dk) #[batch, input_shape, input_shape]
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim = -1)
    if dropout!=0:
        F.dropout(attn, p=dropout, training=train_status, inplace=True)
    r = torch.matmul(attn, v)
    r = r.squeeze() #[batch, input_shape]
    return r

class AttentionModel(nn.Module):
    def __init__(self, input_shape, head_num = 1):
        super(AttentionModel, self).__init__()
        self.input_shape = input_shape
        assert self.input_shape % head_num == 0, "input_shape can't split by head_num"
        self.head_num = head_num
        self.wq = nn.Linear(input_shape, input_shape)
        self.wk = nn.Linear(input_shape, input_shape)
        self.wv = nn.Linear(input_shape, input_shape)
        self.concat_layer = nn.Linear(input_shape, input_shape)
    def forward(self, q, k, v):
        b = q.size(0)
        head = self.head_num
        q = self.wq(q)#[batch_size, input_shape]
        k = self.wk(k)
        v = self.wv(v)
        #multi head
        if head> 1:
            dk = self.input_shape//head
            q = q.view(b, head, dk)#[batch_size,head_num,dk]
            k = k.view(b, head, dk)
            v = v.view(b, head, dk)
            r = attention(q, k, v, train_status=self.training, dropout=0, mask=None) #[batch_size, head_num, dk]
            r = r.view(b, self.input_shape)
            r = self.concat_layer(r)
        else:
            r = attention(q, k, v, train_status=self.training, dropout=0, mask=None)
        return r

##############################
#        Feed Forward Layer
##############################
class FeedForwardLayer(nn.Module):
    def __init__(self, input_shape, output_shape, activate = None, norm_layer = None, dropout = 0.):
        super(FeedForwardLayer, self).__init__()
        self.layer = nn.Linear(input_shape, output_shape)
        self.dropout = nn.Dropout(p=dropout)   #输入dropout率
        if norm_layer is None:
            self.norm_layer=None
        else:
            self.norm_layer = norm_layer(output_shape) #输入标准化方法， 如nn.LayerNorm
        self.activate = activate  #输入激活函数实例，如nn.Relu(inplace = True)
    def forward(self, x):
        if self.activate is not None:
            x = self.activate(x)
        x = self.layer(self.dropout(x))
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x
##############################
#        Residual Connection
##############################
class ResConnect(nn.Module):
    def __init__(self, size, dropout = 0.1):
        super(ResConnect, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
##############################
#        Transformer Encoder
##############################
class TransformerEncoder(nn.Module):
    def __init__(self, size, ff_size, head = 1):
        super(TransformerEncoder, self).__init__()
        self.size = size
        self.attn = AttentionModel(size, head_num=head)
        self.connect = utils.clones(ResConnect(size, dropout=0.1), 2)
        ff_list = [
            FeedForwardLayer(size, ff_size),
            FeedForwardLayer(ff_size, size,
                             activate=nn.ReLU(inplace=True),
                             norm_layer=None, dropout=0)
        ]
        self.ffLayer = nn.Sequential(*ff_list)
        self.norm = nn.LayerNorm(size)
    def forward(self, x):
        x = self.connect[0](x, lambda x: self.attn(x, x, x))
        return self.norm(self.connect[1](x, self.ffLayer))
##############################
#        Transformer Decoder
##############################
class TransformerDecoder(nn.Module):
    def __init__(self, size, ff_size, head = 1):
        super(TransformerDecoder, self).__init__()
        self.size = size
        self.self_attn = AttentionModel(size, head_num=head)
        self.cross_attn = AttentionModel(size, head_num=head)
        self.connect = utils.clones(ResConnect(size, dropout=0.1), 3)
        ff_list = [
            FeedForwardLayer(size, ff_size),
            FeedForwardLayer(ff_size, size,
                             activate=nn.ReLU(inplace=True),
                             norm_layer=None, dropout=0)
        ]
        self.ffLayer = nn.Sequential(*ff_list)
        self.norm = nn.LayerNorm(size)
    def forward(self, x, memory):
        x = self.connect[0](x, lambda x: self.self_attn(x, x, x))
        x = self.connect[1](x, lambda x: self.cross_attn(x, memory, memory))
        return self.norm(self.connect[2](x, self.ffLayer))

##############################
#        Transformer Model
##############################
class Transformer(nn.Module):
    def __init__(self, input_size, head_number = 1, reverse = False):
        super(Transformer, self).__init__()
        input_layer_list = [
            FeedForwardLayer(input_size, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, 1024, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
        ]#[Linear+Norm+Relu+Drop+Linear]  enc[Norm+...]
        output_layer_list = [
            FeedForwardLayer(1024, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, input_size, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
            nn.Softplus()
        ]#dec[...+Norm]  [Linear+Norm+ReLu+Drop+Linear+SoftPlus]
        self.input_layer = utils.clones(nn.Sequential(*input_layer_list),2)
        self.enc = TransformerEncoder(1024, 2048, head = head_number)
        self.dec = TransformerDecoder(1024, 2048, head = head_number)
        self.output_layer = nn.Sequential(*output_layer_list)
        self.reverse = reverse
    def forward(self, source, target):
        source_emb = self.input_layer[0](source)
        target_emb = self.input_layer[1](target)
        if self.reverse:
            output = self.output_layer(self.dec(source_emb, self.enc(target_emb)))
        else:
            output = self.output_layer(self.dec(target_emb, self.enc(source_emb)))
        return output
    def encode(self, source, target):
        if self.reverse:
            enc_out = self.enc(self.input_layer[1](target))
        else:
            enc_out = self.enc(self.input_layer[0](source))
        return enc_out
    def decode(self, source, target):
        source_emb = self.input_layer[0](source)
        target_emb = self.input_layer[1](target)
        if self.reverse:
            dec_out = self.dec(source_emb, self.enc(target_emb))
        else:
            dec_out = self.dec(target_emb, self.enc(source_emb))
        return dec_out
#多层编码解码的Transformer
class TransformerMultiLayer(nn.Module):
    def __init__(self, input_size, head_number = 1, n_layers = 1):
        super(TransformerMultiLayer, self).__init__()
        input_layer_list = [
            FeedForwardLayer(input_size, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, 2048, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
        ]#[Linear+Norm+Relu+Drop+Linear]  enc[Norm+...]
        output_layer_list = [
            FeedForwardLayer(2048, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, input_size, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
            nn.Softplus()
        ]#dec[...+Norm]  [Linear+Norm+ReLu+Drop+Linear+SoftPlus]
        #在此处编辑编码器层数
        self.enc_list = utils.clones(TransformerEncoder(2048,1024, head = head_number),n_layers)
        self.dec_list = utils.clones(TransformerDecoder(2048,1024, head=head_number), n_layers)
        self.input_layer = utils.clones(nn.Sequential(*input_layer_list),2)
        self.output_layer = nn.Sequential(*output_layer_list)
    def forward(self, source, target):
        source_emb = self.input_layer[0](source)
        target_emb = self.input_layer[1](target)
        for enc in self.enc_list:
            source_emb = enc(source_emb)
        for dec in self.dec_list:
            target_emb = dec(target_emb, source_emb)
        output = self.output_layer(target_emb)
        return output
    def encode(self, source):
        out = self.input_layer[0](source)
        for enc in self.enc_list:
            out = enc(out)
        return out
    def decode(self, source, target):
        enc_out = self.input_layer[0](source)
        dec_out = self.input_layer[1](target)
        for enc in self.enc_list:
            enc_out = enc(enc_out)
        for dec in self.dec_list:
            dec_out = dec(dec_out, enc_out)
        return dec_out


class BERT(nn.Module):
    def __init__(self, input_size, head_number = 1, n_layers = 1):
        super(BERT, self).__init__()
        input_layer_list = [
            FeedForwardLayer(input_size, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, 1024, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
        ]  # [Linear+Norm+Relu+Drop+Linear]  enc[Norm+...]
        output_layer_list = [
            FeedForwardLayer(1024, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, input_size, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
            nn.Softplus()
        ]  # dec[...+Norm]  [Linear+Norm+ReLu+Drop+Linear+SoftPlus]
        enc_layer = []
        self.input_layer = nn.Sequential(*input_layer_list)
        for _ in range(n_layers):
            enc_layer += [TransformerEncoder(1024, 2048, head=head_number)]
        self.enc = nn.Sequential(*enc_layer)
        self.output_layer = nn.Sequential(*output_layer_list)
    def forward(self, x):
        return self.output_layer(self.enc(self.input_layer(x)))
    def encode(self, x):
        return self.enc(self.input_layer(x))
