from encoder_decoder import *
import torch 

# Citation: 
# https://github.com/SantoshDhirwani/stick_breaking_vae/blob/master/model_classes/VAEs_pytorch.py#L28

class sb_vq_vae(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_latent, beta, prior_alpha, prior_beta, training=True):
        super(sb_vq_vae, self).__init__()
        self.encoder = encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = InnerProductDecoder()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_latent = num_latent
        self.training = training
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.beta = beta



        self.encoder = encoder(input_dim, hidden_dim, latent_dim)
        self.sb_vq_layer = sb_vq_layer(latent_dim, num_latent, training)
        self.innerproductdecoder = InnerProductDecoder()


## Reconstruction loss for whole graph (scaled based on # or positive edges)
    def recon_loss(self, quantized_embedding, pos_edge_index, neg_edge_index=None, sigmoid=True):
        tol = 1e-15
        pos_loss = -torch.log(self.innerproductdecoder(quantized_embedding, pos_edge_index, sigmoid) + tol).mean()
        num_pos = pos_edge_index.shape[1]
        # To make the experiments reproducible
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, num_neg_samples=quantized_embedding.size(0))
        neg_loss = -torch.log(1 -
                            self.innerproductdecoder(quantized_embedding, neg_edge_index, sigmoid) +
                            tol).mean()
        return pos_loss  + neg_loss  # * pos_edge_index.shape[1] / (pos_edge_index.shape[1] + neg_edge_index.shape[1])

# Sample
    def sample(self):
        num = 50
        pi = torch.zeros(num, self.num_latent)
        torch.manual_seed(2000)
        v = torch.distributions.Beta(concentration1=self.prior_alpha,concentration0=self.prior_beta).sample_n(self.num_latent*num).reshape(num,self.num_latent)
        pi = torch.zeros_like(v)
        for k in range(self.num_latent):
            if k == 0:
                pi[:, k] = v[:, k]
            else:
                pi[:, k] = v[:, k] * torch.stack([(1 - v[:, j]) for j in range(self.num_latent-1) if j < k]).prod(axis=0)
        pi = torch.clip(pi, min=0.001, max=0.999)
        return pi # [num * K]


# The kl for one node
    def kl_prior1_loss(self, dist_probs, pi, x):
        kl_sum = torch.tensor([0.0])
        num = 50
        for i in range(num):
            kl = dist_probs * (dist_probs.log() - torch.log(pi[i,]))
            kl[(dist_probs == 0).expand_as(kl)] = 0
            kl_sum += kl.sum()
        return kl_sum / num 
    
    # forward
    def forward(self, x, pos_edge_index, neg_edge_index=None):
        latent_embedding = self.encoder(x, pos_edge_index)
        quantized_latents, codebook, dist_probs, vq_loss= self.sb_vq_layer(latent_embedding)  
        # print('v:', v)
        # print('codebook:', codebook)
        recon_loss = self.recon_loss(quantized_latents, pos_edge_index, neg_edge_index, sigmoid=True)
        pi = self.sample()
        kl_prior1_loss = self.kl_prior1_loss(dist_probs, pi, x)
        # print('kl_prior1_loss', kl_prior1_loss)
        # print('recon_loss', recon_loss)
        # print('vq_loss', vq_loss)
        # print('kl_cat_loss', kl_cat_loss)
        return latent_embedding, quantized_latents, codebook, recon_loss, kl_prior1_loss, dist_probs, pi, vq_loss




