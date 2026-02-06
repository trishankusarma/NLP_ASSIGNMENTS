import torch
import torch.nn as nn
import numpy as np

"""Skip-gram Word2Vec model implementation"""
class SkipGramModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input embeddings (center word)
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output embeddings (context word)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self.in_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.out_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)

    def forward(self, center_words, context_words, negative_samples):
        """
        Args:
            center_words: tensor of shape (batch_size,)
            context_words: tensor of shape (batch_size,)
            negative_samples: tensor of shape (batch_size, num_negative_samples)
        Returns:
            loss: scalar tensor
        """
        batch_size = center_words.size(0)
        
        # Get embeddings
        center_embeds = self.in_embeddings(center_words)  # (batch_size, embedding_dim)
        context_embeds = self.out_embeddings(context_words)  # (batch_size, embedding_dim)
        neg_embeds = self.out_embeddings(negative_samples)  # (batch_size, num_neg, embedding_dim)
        
        eps = 1e-10
        # Positive score
        # dot product of center_words and context words 
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # (batch_size,) 
        # Goal: Maximize probability that context word appears near center word :: So Minimize :: -log( sigmoid(pos_score) + eps )
        pos_loss = -torch.log(torch.sigmoid(pos_score) + eps) 
                
        # Negative scores
        # neg_embeds = (b, n, k) and center_embeds = (b, k) -> center_embeds.unsqueeze(2) -> (b, k, 1)
        # bmm -> batch dot product -> (b, n, 1) -> squeeze -> (b, n)
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)  # (batch_size, num_neg)
        # Goal: Maximize probability that negative words DON'T appear near center word :: So Minimize -log(sigmoid(-neg_score) + eps )
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + eps), dim=1)
        
        # Total loss
        loss = torch.mean(pos_loss + neg_loss)
        return loss
    
    def get_embedding(self, word_idx):
        """Get embedding for a word index"""
        return self.in_embeddings.weight[word_idx].detach().cpu().numpy()