import torch
import torch.nn as nn

# Model Architecture
class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTM, self).__init__()
        num_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional=True,
            batch_first=True
        )

        self.out = nn.Linear(512, 1)

    def forward(self, x):
        # [8, 128]
        x = self.embedding(x)
        # [8, 128, 100]
        x, _ = self.lstm(x)
        # [8, 128, 256]
        avg_pool = torch.mean(x, 1)
        # [8, 256]
        max_pool, _ = torch.max(x, 1)
        # [8, 256]
        out = torch.cat((avg_pool, max_pool), 1)
        # [8, 512]
        out = self.out(out)
        # [8, 1]

        return out