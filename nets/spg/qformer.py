from .med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()
from .vqvae_modules import VectorQuantizerEMA, ConvNormRelu, Res_CNR_Stack

import torch
from torch import nn
import torch.nn.functional as F

from .blip import create_vit, init_tokenizer, load_checkpoint
import math
import torch.nn.init as init

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: B x T x d_model tensor
        :return: B x T x d_model tensor
        """
        x = x + self.pe[None, : x.shape[1], :]
        x = self.dropout(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, in_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(AudioEncoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.project = ConvNormRelu(in_dim, self._num_hiddens // 4, leaky=True)

        self._enc_1 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)
        self._down_1 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 2, leaky=True, residual=True,
                                    sample='down')
        self._enc_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True)
        self._down_2 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens, leaky=True, residual=True)
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)

    def forward(self, x, frame_num=0):
        h = self.project(x)  
        h = self._enc_1(h)
        h = self._down_1(h)
        h = self._enc_2(h)
        h = self._down_2(h)
        h = self._enc_3(h)
        return h



            

        
class qformer(nn.Module):
    def __init__(self,                  
                 batchsize = 128 ,
                 q_lenth = 44,
                 width = 768, 
                 embed_dim = 512,     
                 codebook_size = 2048,
                 num_layers = 6,
                 random = True,
                 num_q = 1
                 ):
        super().__init__()

        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=512,nhead=8,batch_first=True,activation="relu")
        self.position = PositionalEncoding(embed_dim, dropout=0)
        self.text_encoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer,num_layers=num_layers)
        self.audioencoder = AudioEncoder(in_dim=128,num_hiddens=512,num_residual_layers=2,num_residual_hiddens=0)
        self.width = width
        self.proj = nn.Linear(512,codebook_size)
        self.apply(self.weights_init) 
        self.num_q = num_q

    def weights_init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.01)

    def get_tgt_mask(self, size: int, device: str) -> torch.tensor:
        mask = torch.tril(
            torch.ones((size, size), device=device) == 1
        )  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask
    
    def forward(self,audio_feat,video_input,ablation=False):
        if not ablation:
            b,seq_len = audio_feat.shape[0],audio_feat.shape[1]
            audio_feat = self.audioencoder(audio_feat[:,:].transpose(1,2),frame_num = 0).transpose(1,2)### bx seqx 512
            inputs_embeds = audio_feat
            if len(video_input.shape) == 2:
                video_input = video_input.unsqueeze(0)
            audio_embeds = self.position(inputs_embeds)
            #print(audio_embeds.shape)
            tgt_mask = self.get_tgt_mask(audio_embeds.shape[1], audio_embeds.device)
            video_input = video_input.repeat(1,seq_len//2,1) 
            video_embeds = self.position(video_input) 
            out_put = self.text_encoder(tgt=audio_embeds,memory = video_embeds,tgt_mask=tgt_mask)
            projhead = self.proj(out_put)
            try:
                prjhead = projhead.view(b,-1,2,2048)
            except:
                prjhead = projhead[:,:-1,:].view(b,-1,2,2048)
        return prjhead

    def generate(self,shape=(8,8),batch_size=64,aud_feat=None,text_feat=None): ### 22x2

        param = next(self.parameters())

        shape[0] = shape[0]//4

        x = torch.zeros(
            (batch_size,*shape),
            dtype=torch.int64,device=param.device
        )

        h0 = 0
        print(x.shape)
        h = shape[0] 
        print(shape)
        for i in range(h0,h):
            for j in range(shape[1]):
                logits = self.forward(aud_feat,text_feat,ablation=False).permute(0, 3, 1, 2).contiguous()
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x[:, h0:h]

    
        
    
   
    
    
    