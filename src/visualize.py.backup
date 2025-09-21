import torch

def aggregate_attention(attentions):
    attn = [a.mean(dim=1) for a in attentions]
    cls_attn = [a[:, 0, :] for a in attn]
    scores = torch.mean(torch.stack(cls_attn), dim=0)
    scores = scores / scores.max(dim=1, keepdim=True).values
    return scores.detach().cpu().numpy()

def render_attention(tokens, scores):
    html_output = ""
    for token, score in zip(tokens, scores):
        color = f"rgba(255, 0, 0, {score:.2f})"
        html_output += f"<span style='background-color:{color}; padding:2px;'>{token.replace('##','')} </span>"
    return html_output

def show_attention(text, tokenizer, attentions, inputs):
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    scores = aggregate_attention(attentions)[0]
    return render_attention(tokens, scores)
