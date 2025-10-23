import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, cache_dir: str | None = None, quantize: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=quantize,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir=cache_dir,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer


def get_embedding_matrix(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    return model.get_input_embeddings().weight


def generate(model, input_embeddings, num_tokens: int, temperature: float = 0.0):
    embedding_matrix = get_embedding_matrix(model)
    current_embeddings = input_embeddings.clone()
    generated_tokens: list[int] = []

    with torch.no_grad():
        for _ in range(num_tokens):
            outputs = model(inputs_embeds=current_embeddings)
            next_token_logits = outputs.logits[:, -1, :]
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1)
            else:
                next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(-1)

            generated_tokens.append(next_token.item())
            next_token_embedding = embedding_matrix[next_token].unsqueeze(0)
            current_embeddings = torch.cat((current_embeddings, next_token_embedding), dim=1)

    return generated_tokens, current_embeddings


def calc_loss(model, embeddings, embeddings_target, targets):
    full_embeddings = torch.cat([embeddings, embeddings_target], dim=1)
    outputs = model(inputs_embeds=full_embeddings)
    logits = outputs.logits
    loss_start = embeddings.shape[1] - 1
    loss = nn.CrossEntropyLoss()(logits[:, loss_start:-1, :].reshape(-1, logits.size(-1)), targets)
    return loss, logits


def get_embeddings(tokens, embed_weights, device):
    one_hot = torch.zeros((len(tokens), embed_weights.size(0)), dtype=torch.bfloat16, device=device)
    one_hot[range(len(tokens)), tokens] = 1.0
    embeddings = one_hot @ embed_weights
    embeddings = embeddings.unsqueeze(0)
    return embeddings


def get_one_hot_and_embeddings(tokens, embed_weights, device):
    one_hot = torch.zeros((len(tokens), embed_weights.size(0)), dtype=torch.bfloat16, device=device)
    one_hot[range(len(tokens)), tokens] = 1.0
    embeddings = one_hot @ embed_weights
    embeddings = embeddings.unsqueeze(0)
    return one_hot, embeddings