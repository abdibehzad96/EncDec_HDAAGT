
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *


class Positional_Encoding_Layer(nn.Module):
    def __init__(self, hidden_size, num_att_heads, xy_indx, pos_embedding_dim, pos_embedding_dict_size, nnodes, sl, dropout = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.xy_indx = xy_indx
        self.position_embedding = nn.Embedding(sl+3, hidden_size)
        # Embeddings of the positional features
        self.Postional_embeddings = nn.ModuleList()
        for i in range(len(pos_embedding_dim)):
            self.Postional_embeddings.append(nn.Embedding(pos_embedding_dict_size[i], pos_embedding_dim[i], padding_idx=0))

        # Temporal Convolution and Positional Attention layers
        self.TemporalConv = TemporalConv(hidden_size, sl)
        self.Position_Att = nn.MultiheadAttention(embed_dim= hidden_size, num_heads=num_att_heads, batch_first=True)
        self.Pos_FF = FeedForwardNetwork(d_model= hidden_size, out_dim=hidden_size)
        self.Position_Rezero = nn.Parameter(torch.zeros(hidden_size))


        # The recent results showed that placing DAAG layer here is more efficient than putting it after the Encoder
        self.DAAG = DAAG_Layer(in_features=hidden_size, out_features=hidden_size, n_heads= num_att_heads, n_nodes = nnodes, concat=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Scene, Scene_mask, adj):
        B, SL, Nnodes, _ = Scene.size()

        # Embedding the positional features
        positional_embedding = []
        for n, embedding in enumerate(self.Postional_embeddings):
            positional_embedding.append(embedding(Scene[...,n].long()))
        Temporal_embd = torch.cat(positional_embedding, dim=-1)

        # DAAG layer
        Spatial_embd = self.DAAG(Temporal_embd, adj) 
        # Pos_embd = Pos_embd.permute(0,2,1,3).reshape(B*Nnodes, SL, 2*self.hidden_size)

        # Leaky_Residual
        # Pos_embd= torch.cat((Pos_embd,Leaky_residual), dim=-1)

        # Positional Encoding + Attention
        Temporal_embd = Temporal_embd.permute(0,2,1,3).reshape(B*Nnodes, SL, self.hidden_size)
        Spatial_embd = Spatial_embd.permute(0,2,1,3).reshape(B*Nnodes, SL, self.hidden_size)
        Temporal_embd = Temporal_embd + self.TemporalConv(Temporal_embd)
        
        # Temporal_embd = Temporal_embd + positional_encoding(Temporal_embd, self.hidden_size)
        Temporal_embd = Temporal_embd + self.position_embedding(torch.arange(0, SL, device=Scene.device)).unsqueeze(0)
        Temporal_embd_att = self.Position_Att(Temporal_embd , Temporal_embd , Temporal_embd, need_weights=False, key_padding_mask = Scene_mask)[0] #key_padding_mask = src_mask
        Temporal_embd = self.Position_Rezero*Temporal_embd_att + Temporal_embd
        Temporal_embd = self.dropout(Temporal_embd)
        return Temporal_embd, Spatial_embd, 0


class Mixed_Attention_Layer(nn.Module):
    def __init__(self, hidden_size, num_att_heads):
        super().__init__()
        self.Mixed_Att = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_att_heads, batch_first=True)
        self.Mixed_FF = FeedForwardNetwork(d_model=hidden_size, out_dim=hidden_size)
        self.Mixed_Rezero = nn.Parameter(torch.zeros(hidden_size))
        self.Mixed_Rezero2 = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, Temporal_embd, Spatial_embd, Scene_mask):
        # Cross Attention
        Att_mixed = self.Mixed_Att(Temporal_embd, Spatial_embd, Spatial_embd, key_padding_mask = Scene_mask, need_weights=False)[0] #key_padding_mask = src_mask
        Mixed = self.Mixed_Rezero*Att_mixed + Temporal_embd
        Mixed_FF = self.Mixed_FF(Mixed)
        Mixed = Mixed + self.Mixed_Rezero2*Mixed_FF
        Mixed = self.dropout(Mixed)
        return Mixed

class Encoder_DAAG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        self.xy_indx = config['xy_indx']
        self.Positional_Encoding_Layer = Positional_Encoding_Layer(hidden_size = self.hidden_size, num_att_heads = num_heads, 
                                                                   xy_indx = self.xy_indx, pos_embedding_dim = config['pos_embedding_dim'],
                                                                   pos_embedding_dict_size = config['pos_embedding_dict_size'], nnodes = config['Nnodes'], sl = config['sl']//config['dwn_smple'])
        self.Mixed_Att_Layer = Mixed_Attention_Layer(hidden_size = self.hidden_size, num_att_heads = num_heads)

    def forward(self, Scene, Scene_mask, Adj_mat):
        Temporal_embd, Spatial_embd, _ = self.Positional_Encoding_Layer(Scene, Scene_mask, Adj_mat)
        # Traffic_embd = self.Traffic_Encoding_Layer(Scene, Scene_mask, Leaky_Res)
        Temporal_embd = self.Mixed_Att_Layer(Temporal_embd, Spatial_embd, Scene_mask)
        return Temporal_embd


class Projection(nn.Module):
    def __init__(self, hidden_size, output_size, output_dict_size):
        super(Projection, self).__init__()
        self.output_dict_size = output_dict_size
        # self.LN = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size*output_dict_size)
        self.output_size = output_size
        
    def forward(self, x):
        BN, SL, _ = x.size()
        # x = self.LN(x)
        x = self.linear2(x)
        return x.reshape(BN, SL, self.output_size, self.output_dict_size)


class HDAAGT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        output_size = len(config['xy_indx'])
        Nnodes = config['Nnodes']
        self.num_heads = config['num_heads']
        self.future_len = config['future']// config['dwn_smple']
        self.encoder = Encoder_DAAG(config)
        self.decoder = Decoder(self.hidden_size,config['num_heads'], config['pos_embedding_dim'], config['pos_embedding_dict_size'], config['xy_indx'], config['dropout'])
        self.proj = Projection(self.hidden_size, output_size, output_dict_size= config['output_dict_size'])
    def forward(self, scene: torch.Tensor, src_mask, adj_mat: torch.Tensor, target):
        F = target.size(1)
        trg = target.permute(0,2,1,3).reshape(-1,F,2)  # Remove the eos token for the input to the decoder
        trg_mask = target_mask(trgt=trg, num_head=self.num_heads, device=scene.device)
        enc_out = self.encoder(scene, src_mask, adj_mat)
        dec_out = self.decoder(trg, enc_out, trg_mask, src_mask)
        proj = self.proj(dec_out)
        return proj


class DecoderLayer(nn.Module):

    def __init__(self, d_model, hidden_size, num_att_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(embed_dim= hidden_size, num_heads=num_att_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = nn.MultiheadAttention(embed_dim= hidden_size, num_heads=num_att_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = FeedForwardNetwork(d_model= hidden_size, out_dim=hidden_size)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, trgt, enc, trg_mask, src_mask): #, src_mask):
        # 1. compute self attention
        _x = trgt
        x = self.self_attention(trgt, trgt, trgt, attn_mask= trg_mask, need_weights=False)[0]  #, key_padding_mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(query=x, key=enc, value=enc, key_padding_mask=src_mask, need_weights=False)[0]

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, hidden_size,num_att_heads, pos_embedding_dim, pos_embedding_dict_size, xy_indx, drop_prob, max_len=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.xy_indx = xy_indx
        # Embeddings of the positional features
        self.Postional_embeddings = nn.ModuleList()
        for i in xy_indx:
            self.Postional_embeddings.append(nn.Embedding(pos_embedding_dict_size[i], hidden_size//2, padding_idx=0))

        # The recent results showed that placing DAAG layer here is more efficient than putting it after the Encoder
        # self.DAAG = DAAG_Layer(in_features=hidden_size, out_features=hidden_size, n_heads= num_att_heads, n_nodes = nnodes, concat=True)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.layer = DecoderLayer(hidden_size, hidden_size, num_att_heads, drop_prob)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = self.emb(trg)
        B, SL, _ = trg.size()

        # Embedding the positional features
        positional_embedding = []
        for n, embedding in enumerate(self.Postional_embeddings):
            positional_embedding.append(embedding(trg[...,n].long()))
        Temporal_embd = torch.cat(positional_embedding, dim=-1)
       

        positions = torch.arange(0, SL).unsqueeze(0).to(trg.device)
        Temporal_embd = Temporal_embd + self.position_embedding(positions)

        # for layer in self.layers:
        output = self.layer(Temporal_embd, enc_src, trg_mask, src_mask)

        # pass to LM head
        # output = self.linear(trg)
        return output


if __name__ == "__main__":
    print("This is a module file, not meant to be run directly.")