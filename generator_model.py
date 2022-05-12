import torch
from torch.distributions.categorical import Categorical
from torch.distributions import kl_divergence

from model_center.model import CPM1


class GeneratorModel(torch.nn.Module):

    def __init__(self, cpm_path):
        super().__init__()
        self.generator_model = CPM1.from_pretrained(cpm_path)
        generator_dim = 1024
        self.dynamic_score_projection = torch.nn.Sequential(
            torch.nn.Linear(in_features=generator_dim, out_features=generator_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=generator_dim, out_features=generator_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=generator_dim, out_features=1),
        ).cuda()

    def forward(self, context_input_ids, doc_scores, labels, tokenizer):
        with torch.no_grad():
            labels_input = torch.cat([labels.unsqueeze(0) for _ in range(context_input_ids.shape[0])], dim=0)
            input_tokens = torch.cat((context_input_ids, labels_input), dim=1)
            input_context = torch.cat((torch.ones_like(context_input_ids), torch.zeros_like(labels_input)), dim=1)
            input_span = torch.zeros_like(input_context)

        # forwarding in cpm model
        logits, hidden_states = self.generator_model(input_tokens, torch.Tensor([input_tokens.shape[1]] * input_tokens.shape[0]).cuda(), input_context, input_span)
        logits = logits[:, context_input_ids.shape[1]:, :].float()  # (top_k, max_target_len, vocab_size)
        hidden_states = hidden_states[:, context_input_ids.shape[1]:, :]  # (top_k, max_target_len, d_model)
        dynamic_scores = self.dynamic_score_projection(hidden_states.float()).squeeze(2)  # (top_k, max_target_len)

        # shift left
        labels = torch.cat([labels[1:], labels.new(1).fill_(tokenizer.pad_id)])

        # calc generation loss
        seq_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)  # (top_k, max_target_len, vocab_size)
        doc_logprobs = torch.log_softmax(dynamic_scores, dim=0)  # (top_k, max_target_len)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1)  # (top_k, max_target_len, vocab_size)
        logprobs = torch.logsumexp(log_prob_sum, dim=0)  # (max_target_len, vocab_size)
        labels = labels.unsqueeze(-1)  # (max_target_len, 1)
        generation_loss = logprobs.gather(dim=-1, index=labels)  # (max_target_len, 1)
        generation_loss.masked_fill_(labels.eq(tokenizer.pad_id), 0.0)
        generation_loss = -generation_loss.sum(0).squeeze(0)

        # calc consistency loss
        g_probs = torch.softmax(dynamic_scores, dim=0).mean(1)
        g_probs = g_probs.detach()
        r_probs = torch.softmax(doc_scores, dim=0).clamp(min=1e-5)
        consistency_loss = kl_divergence(Categorical(probs=g_probs), Categorical(probs=r_probs))

        return generation_loss, consistency_loss

    @torch.no_grad()
    def generate(self, context_input_ids, tokenizer, max_length):
        preds = torch.IntTensor([1]).cuda()
        while preds.shape[0] < max_length:
            labels_input = torch.cat([preds.unsqueeze(0) for _ in range(context_input_ids.shape[0])], dim=0)
            input_tokens = torch.cat((context_input_ids, labels_input), dim=1)
            input_context = torch.cat((torch.ones_like(context_input_ids), torch.zeros_like(labels_input)), dim=1)
            input_span = torch.zeros_like(input_context)

            # forwarding in cpm model
            logits, hidden_states = self.generator_model(input_tokens, torch.Tensor([input_tokens.shape[1]] * input_tokens.shape[0]).cuda(), input_context, input_span)
            logits = logits[:, -1, :].float()  # (top_k, vocab_size)
            hidden_states = hidden_states[:, -1, :]  # (top_k, d_model)
            dynamic_scores = self.dynamic_score_projection(hidden_states.float()).squeeze(1)  # (top_k)

            # calc probability
            seq_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)  # (top_k, vocab_size)
            doc_logprobs = torch.log_softmax(dynamic_scores, dim=0)  # (top_k)
            log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1)  # (top_k, vocab_size)
            logprobs = torch.logsumexp(log_prob_sum, dim=0)  # (vocab_size)
            pred = torch.argmax(logprobs).unsqueeze(0)
            if preds[-1] == tokenizer.pad_id or preds[-1] == tokenizer.eod_id:
                pred = torch.IntTensor([tokenizer.pad_id]).cuda()
            preds = torch.cat((preds, pred))

        return preds
