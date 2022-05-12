import torch
from transformers import BertModel


class RetrieverModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.retriever_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext").cuda()
        retriever_dim = 768
        self.retriever_output = torch.nn.Sequential(
            torch.nn.Linear(in_features=retriever_dim, out_features=retriever_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=retriever_dim, out_features=retriever_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=retriever_dim, out_features=1),
        ).cuda()

    def forward(self, input_ids, attention_mask):
        retriever_outputs = self.retriever_model(input_ids, attention_mask)
        logits = self.retriever_output(retriever_outputs.last_hidden_state)
        return logits
