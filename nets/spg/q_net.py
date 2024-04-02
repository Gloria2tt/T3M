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
        #print('0',h.shape)  
        h = self._enc_1(h)
        #print('1',h.shape)
        h = self._down_1(h)
        #print(h.shape)
        h = self._enc_2(h)
        #print(h.shape)
        h = self._down_2(h)
        #print(h.shape)
        h = self._enc_3(h)
        return h
"""class pose_estimator(nn.Module):
    def __init__(self, config):
        super().__init__()
    def __forward__(self,):
        pass
    pass"""

class qformerEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        #self.audioencoder = AudioEncoder(in_dim=128,num_hiddens=)
        self.config = config

    def forward(
        self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class prompt_learners(nn.Module):
    def __init__(self, batchsize,lenth,dim,random = True):
        super().__init__()
        self.seq_lenth = lenth
        self.width = dim
        self.random = random
        if self.random:
            vectors = torch.empty(batchsize,self.seq_lenth,self.width)
            nn.init.normal_(vectors, std=0.02)
        self.q_vectors = nn.Parameter(vectors)
    
    def forward(self,):
        q_l = self.q_vectors
        
        return q_l
            

        
class qformer_net(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/bert_config.json', 
                 batchsize = 128 ,
                 q_lenth = 44,
                 width = 768, 
                 embed_dim = 512,     
                 codebook_size = 2048,
                 num_layers = 8
                 ):
        super().__init__()
        encoder_config = BertConfig.from_json_file(med_config)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=512,nhead=8,batch_first=True)
        self.position = PositionalEncoding(512, dropout=0.1)
        self.text_encoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer,num_layers=num_layers)
        self.mlp1 = nn.Linear(128,embed_dim)
        #text_with = self.text_encoder.config.hidden_size
        #self.input_mlp = nn.Linear(512,1024)
        #self.input_mlp2 = nn.Linear(1024,codebook_size) 
        #self.input_mlp = nn.Linear(128,embed_dim)
        #self.input_mlp2 = nn.Linear(embed_dim,width)
        #self.embeddings = qformerEmbeddings(config = encoder_config)
        #self.audioencoder = AudioEncoder(in_dim=128,num_hiddens=512,num_residual_layers=2,num_residual_hiddens=0)
        self.width = width
        self.proj = nn.Linear(embed_dim,129)
        
        self.apply(self.weights_init)

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
    
    def forward(self,audio_feat,video_input,ablation=False,max_lenth = 150):
        if not ablation:
            b,seq_len = audio_feat.shape[0],audio_feat.shape[1]
            audio_feat = self.mlp1(audio_feat)
            #audio_feat = self.audioencoder(audio_feat[:,:].transpose(1,2),frame_num = 0).transpose(1,2)### bx seqx 512
            inputs_embeds = audio_feat
            audio_embeds = self.position(inputs_embeds)
            #print(audio_embeds.shape)
            tgt_mask = self.get_tgt_mask(audio_embeds.shape[1], audio_embeds.device)
            #video_input = video_input.repeat(1,seq_len//2,1) ### bx1x512 - b xseq/2 x 512 
            #video_embeds = self.position(video_input) ## 300 - 500 - 300 - 150 diffusion 
            if len(video_input.shape) == 2:
                video_input = video_input.unsqueeze(0)
            #print(video_input.shape)
            video_input = video_input.repeat(1,seq_len//2,1)
            #video_embeds = self.position(video_input)
            #video_embeds = video_input
            video_embeds = self.position(video_input)
            #print(video_embeds.shape)
            out_put = self.text_encoder(tgt=audio_embeds,memory = video_embeds,tgt_mask = tgt_mask)
            #out_put = self.audioencoder(out_put.transpose(1,2)).transpose(1,2) ### bxseqx1024
            #print(out_put.shape)
            #projhead = out_put.view(b,seq_len//4,2,2048)
            #projhead = self.input_mlp(out_put)
            #projhead = self.input_mlp2(projhead)
            prjhead = self.proj(out_put)
            """try:
                prjhead = projhead.view(b,-1,2,2048)
            except:
                prjhead = projhead[:,:-1,:].view(b,-1,2,2048)"""
        else:
            b,seq_len = audio_feat.shape[0],audio_feat.shape[1]
            #print(audio_feat.shape)
            """audio_feat = self.input_mlp(audio_feat)
            audio_feat = self.input_mlp2(audio_feat)
            q_feat = self.q_embeddings()"""
            audio_feat = self.audioencoder(audio_feat[:,:].transpose(1,2)).transpose(1,2)         
            #print('audio shape:------',audio_feat.shape)
            #audio_feat = audio_feat.view(b,-1,audio_feat.shape[-1])
            #print('true last chec',audio_feat.shape,q_feat.shape,video_input.shape)
            #inputs_embeds = torch.cat([audio_feat,q_feat],dim=1)
            inputs_embeds = audio_feat
            padded_input_emb = torch.zeros((b, max_lenth, 768), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            padded_input_emb[:, :seq_len//2, :] = inputs_embeds[:, :, :]
            #padded_input_emb = torch.nn.utils.rnn.pad_sequence(inputs_embeds, batch_first=True, padding_value=0.0, maxlen=max_lenth)
            print(padded_input_emb.shape)
            audio_embeds = self.embeddings(inputs_embeds=padded_input_emb)
            mask = torch.zeros((b, max_lenth), dtype=torch.float32, device=inputs_embeds.device)
            mask[:, :seq_len] = 1.0
            mask = mask.type(torch.float32)
            #audio_atts = torch.ones(audio_embeds.size()[:-1],dtype=torch.long).to(audio_feat.device)
            #attention_mask = torch.ones([b,seq_len])
            #video_feat = video_input.repeat(1,44,1)
            #video_embeds = self.embeddings(inputs_embeds = video_feat)
            #video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(video_input.device)
            """b,seq_len = audio_feat.shape[0],audio_feat.shape[1]
            
            audio_feat = self.input_mlp(audio_feat)
            audio_feat = self.input_mlp2(audio_feat)
            #q_feat = self.q_embeddings()
            #print('true last chec',audio_feat.shape,q_feat.shape,video_input.shape)
            inputs_embeds = torch.cat([audio_feat,q_feat],dim=1)
            audio_embeds = self.embeddings(inputs_embeds=inputs_embeds)
            audio_atts = torch.ones(audio_embeds.size()[:-1],dtype=torch.long).to(audio_feat.device)"""
            #attention_mask = torch.ones([b,seq_len])
            """video_feat = video_input.repeat(1,44,1)
            video_embeds = self.embeddings(inputs_embeds = video_feat)
            video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(video_input.device)"""
            out_put = self.text_encoder(inputs_embeds=audio_embeds,
                                        attention_mask = mask,
                                        return_dict = True,
                                        mode = 'text'
                                        )
            
            return_out = out_put.last_hidden_state[:,:seq_len//2,:]
            
            projhead = self.proj(return_out)
            prjhead = projhead.view(b,-1,2,2048)
            #print(projhead.shape
            #)
        
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
        h = shape[0] ####300
        print(shape)
        for i in range(h0,h):
            for j in range(shape[1]):
                logits = self.forward(aud_feat,text_feat,ablation=False).permute(0, 3, 1, 2).contiguous()
                #print(logits.shape)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x[:, h0:h]
    
    def generate_mse(self,audio_input,text_feat):
        pred_pose = self.forward(audio_input,text_feat)
        return pred_pose
    

    
        
    
   
    
    
    