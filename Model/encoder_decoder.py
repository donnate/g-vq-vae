from turtle import distance, forward
import torch
from torch_geometric.nn import GraphConv as g_conv
from torch.nn import functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import global_mean_pool as g_mean_pool
from torch.distributions import Categorical, RelaxedOneHotCategorical

# Citation: 
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
# https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py
# https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch

class encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(encoder, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.latent_dim = latent_dim
        # Conv layers (freezed for large-scale graph)
        torch.manual_seed(2000)
        self.g_enconv1 = g_conv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.g_enconv2 = g_conv(hidden_dim, latent_dim)
        self.bn2 = torch.nn.BatchNorm1d(latent_dim)
        # Linear layers
        # self.ln = torch.nn.Linear(latent_dim, hidden_dim)
        # self.ln2 = torch.nn.Linear(hidden_dim, latent_dim)
         
    def forward(self, x, edge_index):
        hidden_embedding = self.g_enconv1(x, edge_index)
        hidden_embedding = self.bn1(hidden_embedding)
        latent_embedding = self.g_enconv2(hidden_embedding, edge_index)
        latent_embedding = self.bn2(latent_embedding)
        # latent_embedding = self.ln(latent_embedding)
        # latent_embedding = latent_embedding.relu()
        # latent_embedding = self.ln2(latent_embedding)
        # latent_embedding.relu()
        # print('latent_embedding:',latent_embedding)
        return latent_embedding




# sb_vq_layer
class sb_vq_layer(torch.nn.Module):
    def __init__(self, latent_dim, num_latent, beta, training=True):
        super(sb_vq_layer, self).__init__()
        self.latent_dim = latent_dim 
        self.num_latent = num_latent 
        self.beta = beta
        self.training = training

        torch.manual_seed(2000)
        self.codebook = torch.nn.Parameter(torch.Tensor(self.num_latent,  self.latent_dim))
        torch.nn.init.uniform_(self.codebook, -1/num_latent, 1/num_latent)

    
    def lgpi2v(self, logpi):
        logv = torch.zeros_like(logpi)
        for k in range(self.num_latent):
            if k == 0 :
                logv[:,k] = logpi[:,k]
            else :
                logv[:,k] = logpi[:,k] - torch.stack([ (1-logv[:,j].exp()).log() for j in range(self.num_latent) if j < k]).sum(axis=0)
        v = torch.exp(logv)
        return v
    
    def pi2v(self, pi):
        # print(pi)
        pi = torch.clip(pi, min = 0.001, max = 0.999)
        pi = pi/ torch.sum(pi, axis=1, keepdim=True)
        # print('pi', pi)
        # print('pi-min', torch.min(pi))
        v = torch.ones_like(pi)
        for k in range(self.num_latent-1):
            if k == 0:
                v[:,k] = pi[:,k]
            else:
                v[:,k] = pi[:,k] / torch.stack([1-v[:,j] for j in range(self.num_latent-1) if j < k]).prod(axis=0)
        # print('v', v)
        # print('v-min', torch.min(v))
        return v



    def forward(self, latent_embedding):
        # latent_embedding [N * D]
        # codebook [K * D]
        # [N*1] + [1*K] - [N*K]
        # distances = (torch.sum(latent_embedding ** 2, dim=1, keepdim=True)) +  \
        #       torch.sum(self.codebook ** 2, dim=1) - \
        #         2 * torch.matmul(latent_embedding, self.codebook.t())
        distances = (torch.sum(latent_embedding ** 2, dim=1, keepdim=True)).detach() +  \
              torch.sum(self.codebook ** 2, dim=1) - \
                2 * torch.matmul(latent_embedding.detach(), self.codebook.t())
        # p(z|eta, x, v) 
        # test the explosion gradient is cause by pi / gumble_softmax
        # print(pi)
        # print(pi.log())
        # print('logits:', logits)
        encoding_inds = torch.argmin(distances, dim=1).unsqueeze(1) # [N] --> [N*1]
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.num_latent) # [N*K]
        encoding_one_hot.scatter_(1, encoding_inds, 1) # [N*K] one-hot vector each row
        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.codebook) # [N*D]
        # VQ Losses sum version
        commitment_loss = F.mse_loss(quantized_latents.detach(), latent_embedding) * self.num_latent
        embedding_loss = F.mse_loss(quantized_latents, latent_embedding.detach()) *  self.num_latent
        vq_loss = embedding_loss + commitment_loss * self.beta
        # probs
        probs = (-distances).softmax(dim=1)
        probs = torch.mean(probs, axis=0, keepdim=True)
        # mean ver2
        # probs = torch.mean(encoding_one_hot, dim=0) + (probs-torch.mean(encoding_one_hot, dim=0)).detach()
        probs = torch.clip(probs, min=0.001, max=0.999)
        # Test
        # print(torch.mean(encoding_one_hot, dim=0))
        return quantized_latents, self.codebook, probs, vq_loss



# decoder for postional information
class InnerProductDecoder(torch.nn.Module):
    def forward(self, quantized_latent_embedding, edge_index, sigmoid=True):
        value = (quantized_latent_embedding[edge_index[0]] * quantized_latent_embedding[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

# decoder for feature
class decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim):
        super(decoder, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.latent_dim = latent_dim
        # Conv layers (freezed for large-scale graph)
        self.g_enconv1 = g_conv(latent_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.g_enconv2 = g_conv(hidden_dim, input_dim)
        # Linear layers
        # self.ln = torch.nn.Linear(latent_dim, hidden_dim)
        # self.ln2 = torch.nn.Linear(hidden_dim, latent_dim)
         
    def forward(self, quantized_f_embedding, edge_index):
        hidden_decoded_embedding = self.g_enconv1(quantized_f_embedding, edge_index)
        hidden_decoded_embedding = self.bn1(hidden_decoded_embedding)
        input_decoded_embedding = self.g_enconv2(hidden_decoded_embedding, edge_index)
        return input_decoded_embedding

class sup_classify(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, number_of_classes):
        super(sup_classify, self).__init__()
        self.hidden_dim = hidden_dim 
        self.latent_dim = latent_dim
        self.number_of_classes = number_of_classes
        self.ln1 = torch.nn.Linear(latent_dim*2, hidden_dim) # *2 because combine p + f 
        self.ln2 = torch.nn.Linear(hidden_dim, number_of_classes)
        
    def forward(self, quantized_p_embedding, quantized_f_embedding):
        quantized_latents = torch.cat([quantized_p_embedding, quantized_f_embedding], dim=1) # [N*2K]
        hidden_embedding = self.ln1(quantized_latents).relu()
        output= self.ln2(hidden_embedding)
        return output


