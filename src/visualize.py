import torch
import numpy as np

def get_content_attention_scores(attentions):
    """Extract meaningful attention scores, excluding special tokens."""
    if not attentions:
        return np.array([])
    
    try:
        # Use last 2 layers for semantic attention
        layer_scores = []
        
        for layer_attention in attentions[-2:]:
            batch_attention = layer_attention[0]  # Remove batch dimension
            avg_heads = batch_attention.mean(dim=0)  # Average across heads
            
            # Get content-to-content attention (exclude CLS and SEP)
            if avg_heads.size(0) > 2:
                content_attention = avg_heads[1:-1, 1:-1]  # Remove first/last tokens
                
                if content_attention.size(0) > 0:
                    token_scores = content_attention.mean(dim=0)
                    layer_scores.append(token_scores)
        
        if layer_scores:
            final_scores = torch.stack(layer_scores).mean(dim=0)
            scores_np = final_scores.detach().cpu().numpy()
            
            # Normalize to [0, 1] for better colors
            if scores_np.max() > scores_np.min():
                scores_np = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min())
            
            return scores_np
            
    except Exception as e:
        print(f"Error in attention extraction: {e}")
        return np.array([])
    
    return np.array([])

def process_tokens_and_scores(tokens, scores):
    """Clean tokens and combine subwords."""
    # Remove special tokens
    if len(tokens) > 2 and tokens[0] == '[CLS]' and tokens[-1] == '[SEP]':
        content_tokens = tokens[1:-1]
    else:
        content_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
    
    # Match lengths
    min_len = min(len(content_tokens), len(scores))
    content_tokens = content_tokens[:min_len]
    scores = scores[:min_len] if len(scores) > 0 else np.zeros(min_len)
    
    # Combine subword tokens
    combined_tokens = []
    combined_scores = []
    current_word = ""
    current_max_score = 0
    
    for token, score in zip(content_tokens, scores):
        clean_token = token.replace('##', '')
        
        if token.startswith('##'):
            current_word += clean_token
            current_max_score = max(current_max_score, score)
        else:
            if current_word:
                combined_tokens.append(current_word)
                combined_scores.append(current_max_score)
            current_word = clean_token
            current_max_score = score
    
    if current_word:
        combined_tokens.append(current_word)
        combined_scores.append(current_max_score)
    
    return combined_tokens, np.array(combined_scores)

def create_attention_html(tokens, scores, min_score=0.1):
    """Create HTML with visible colors."""
    if len(tokens) == 0:
        return "<span>No tokens to display</span>"
    
    html_parts = []
    
    for token, score in zip(tokens, scores):
        if not token.strip():
            continue
        
        # Ensure minimum visibility
        display_score = max(score, min_score)
        
        # Better color scheme with higher opacity
        if display_score > 0.7:
            color = "rgba(220, 38, 38, 0.9)"  # Dark red
            border = "2px solid rgba(185, 28, 28, 1)"
        elif display_score > 0.5:
            color = "rgba(248, 113, 113, 0.8)"  # Medium red  
            border = "1px solid rgba(220, 38, 38, 0.8)"
        elif display_score > 0.3:
            color = "rgba(252, 165, 165, 0.7)"  # Light red
            border = "1px solid rgba(248, 113, 113, 0.6)"
        else:
            color = "rgba(254, 202, 202, 0.6)"  # Very light red
            border = "1px solid rgba(252, 165, 165, 0.4)"
        
        html_parts.append(
            f'<span class="attention-word" '
            f'style="background-color: {color}; '
            f'border: {border}; '
            f'padding: 4px 8px; margin: 2px; border-radius: 4px; '
            f'display: inline-block;" '
            f'title="Score: {score:.3f}">{token}</span>'
        )
    
    return ' '.join(html_parts)

def show_attention(text, tokenizer, attentions, inputs):
    """Main attention visualization function."""
    try:
        all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attention_scores = get_content_attention_scores(attentions)
        
        if len(attention_scores) == 0:
            return f"<span>No attention data for: {text}</span>"
        
        clean_tokens, clean_scores = process_tokens_and_scores(all_tokens, attention_scores)
        
        if len(clean_tokens) == 0:
            return f"<span>No content tokens found</span>"
        
        html_output = create_attention_html(clean_tokens, clean_scores)
        return html_output
        
    except Exception as e:
        print(f"Error in attention visualization: {e}")
        return f"<span style='color: red;'>Error: {str(e)}</span>"