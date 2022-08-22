from encoder_decoder import *
import torch 

# Citation: 
# https://github.com/SantoshDhirwani/stick_breaking_vae/blob/master/model_classes/VAEs_pytorch.py#L28

class sb_vq_vae_f(torch.nn.Module):
    def __init__(self, p_input_dim, f_input_dim, hidden_dim, latent_dim, num_latent, beta, prior_alpha, prior_beta, number_of_classes, training=True):
        super(sb_vq_vae_f, self).__init__()
        self.p_input_dim = p_input_dim
        self.f_input_dim = f_input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_latent = num_latent
        self.training = training
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.beta = beta
        self.number_of_classes = number_of_classes

        self.encoder_pos = encoder(p_input_dim, hidden_dim, latent_dim)
        self.sb_vq_layer_pos = sb_vq_layer(latent_dim, num_latent, training)
        self.encoder_f = encoder(f_input_dim, hidden_dim, latent_dim)
        self.sb_vq_layer_f = sb_vq_layer(latent_dim, num_latent, training)
        self.decoder_f = decoder(latent_dim, hidden_dim, f_input_dim)
        self.innerproductdecoder = InnerProductDecoder()
        self.sup_classify = sup_classify(latent_dim, hidden_dim, number_of_classes)


    ## Reconstruction loss
    def recon_f_loss(self, f, quantized_f_embedding, pos_edge_index):
        input_decoded_embedding = self.decoder_f(quantized_f_embedding, pos_edge_index)
        recon_loss = F.mse_loss(f, input_decoded_embedding)
        return recon_loss

  ## Reconstruction loss for one node on average
    def recon_pos_loss(self, quantized_embedding, pos_edge_index, neg_edge_index=None, sigmoid=True):
        tol = 1e-15
        pos_loss = -torch.log(self.innerproductdecoder(quantized_embedding, pos_edge_index, sigmoid) + tol).mean()
        # To make the experiments reproducible
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, num_neg_samples=quantized_embedding.size(0))
        neg_loss = -torch.log(1 -
                            self.innerproductdecoder(quantized_embedding, neg_edge_index, sigmoid) +
                            tol).mean()
        return (pos_loss + neg_loss) 

# Spare
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


    def kl_prior1_loss(self, dist_probs, pi):
        kl_sum = torch.tensor([0.0])
        num = 50
        for i in range(num):
            kl = dist_probs * (dist_probs.log() - torch.log(pi[i,]))
            kl[(dist_probs == 0).expand_as(kl)] = 0
            kl_sum += kl.sum()
        return kl_sum/num
    
    # forward
    def forward(self, x, f, pos_edge_index, neg_edge_index=None, y=None, mask=None):
        latent_p_embedding = self.encoder_pos(x, pos_edge_index)
        quantized_p_latents, codebook_p, dist_probs_p, vq_loss_p = self.sb_vq_layer_pos(latent_p_embedding) 
        latent_f_embedding = self.encoder_f(f, pos_edge_index) 
        quantized_f_latents, codebook_f, dist_probs_f, vq_loss_f = self.sb_vq_layer_f(latent_f_embedding) 
        recon_p_loss = self.recon_pos_loss(quantized_p_latents, pos_edge_index, neg_edge_index, sigmoid=True)
        recon_f_loss = self.recon_f_loss(f, quantized_f_latents, pos_edge_index)
        pi = self.sample()
        kl_prior_loss_p = self.kl_prior1_loss(dist_probs_p, pi)
        kl_prior_loss_f = self.kl_prior1_loss(dist_probs_f, pi)
        output = self.sup_classify(quantized_p_latents, latent_f_embedding)

        return recon_p_loss, recon_f_loss, kl_prior_loss_p, kl_prior_loss_f, vq_loss_p, vq_loss_f, output


