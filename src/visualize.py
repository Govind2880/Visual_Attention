import torch
import numpy as np

def get_content_attention_scores(attentions):
    """
    Extract attention scores focusing on content tokens, not special tokens.
    
    This addresses the issue where [CLS] and [SEP] tokens dominate attention
    by looking at token-to-token attention within the content.
    """
    if not attentions:
        return np.array([])
    
    try:
        # Use the last few layers which tend to have more semantic attention
        layer_attentions = []
        
        for layer_attention in attentions[-2:]:  # Last 2 layers
            # Shape: [batch, heads, seq_len, seq_len]
            batch_attention = layer_attention[0]  # [heads, seq_len, seq_len]
            
            # Average across attention heads
            avg_heads = batch_attention.mean(dim=0)  # [seq_len, seq_len]
            
            # Method 1: Look at how content tokens attend to each other
            # Exclude first token ([CLS]) and last token ([SEP])
            if avg_heads.size(0) > 2:
                content_region = avg_heads[1:-1, 1:-1]  # Content tokens only
                
                # For each content token, see how much attention it receives from others
                if content_region.size(0) > 0:
                    # Average attention each token receives (incoming attention)
                    token_importance = content_region.mean(dim=0)
                    layer_attentions.append(token_importance)
        
        if layer_attentions:
            # Average across layers
            final_scores = torch.stack(layer_attentions).mean(dim=0)
            
            # Normalize to [0, 1]
            scores_np = final_scores.detach().cpu().numpy()
            if scores_np.max() > 0:
                scores_np = scores_np / scores_np.max()
            
            return scores_np
        
    except Exception as e:
        print(f"Error in content attention extraction: {e}")
    
    # Fallback: use traditional method but still exclude special tokens
    try:
        last_layer = attentions[-1][0]  # [heads, seq_len, seq_len]
        cls_attention = last_layer[:, 0, :].mean(dim=0)  # CLS attention, averaged across heads
        
        # Remove the CLS and SEP attention scores (first and last)
        if len(cls_attention) > 2:
            content_scores = cls_attention[1:-1]  # Exclude first ([CLS]) and last ([SEP])
            
            # Normalize
            scores_np = content_scores.detach().cpu().numpy()
            if scores_np.max() > 0:
                scores_np = scores_np / scores_np.max()
            
            return scores_np
        
    except Exception as e:
        print(f"Error in fallback attention: {e}")
    
    return np.array([])

def clean_tokens_and_scores(tokens, scores):
    """
    Clean tokens by removing special tokens and combining subwords.
    
    Args:
        tokens: List of all tokens from tokenizer
        scores: Attention scores (should match content tokens only)
    
    Returns:
        Tuple of (cleaned_tokens, cleaned_scores)
    """
    # Remove special tokens from the beginning and end
    if len(tokens) > 2 and tokens[0] in ['[CLS]', '<s>'] and tokens[-1] in ['[SEP]', '</s>']:
        content_tokens = tokens[1:-1]
    else:
        content_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '<s>', '</s>']]
    
    # Ensure we have matching lengths
    min_len = min(len(content_tokens), len(scores))
    content_tokens = content_tokens[:min_len]
    scores = scores[:min_len] if len(scores) > 0 else np.zeros(min_len)
    
    # Combine subword tokens
    combined_tokens = []
    combined_scores = []
    current_word = ""
    current_score = 0
    
    for i, (token, score) in enumerate(zip(content_tokens, scores)):
        clean_token = token.replace('##', '')
        
        if token.startswith('##'):
            # This is a subword continuation
            current_word += clean_token
            current_score = max(current_score, score)  # Take the max score for the word
        else:
            # This is a new word
            if current_word:  # Save the previous word
                combined_tokens.append(current_word)
                combined_scores.append(current_score)
            
            current_word = clean_token
            current_score = score
    
    # Don't forget the last word
    if current_word:
        combined_tokens.append(current_word)
        combined_scores.append(current_score)
    
    return combined_tokens, np.array(combined_scores)

def get_attention_color(score, enhanced=True):
    """
    Get color for attention visualization based on score.
    
    Args:
        score: Attention score between 0 and 1
        enhanced: Whether to use enhanced color scheme
    
    Returns:
        CSS color string and additional styling
    """
    if not enhanced:
        return f"rgba(255, 0, 0, {score:.2f})", ""
    
    # Enhanced color scheme with better visual distinction
    if score > 0.8:
        color = "rgba(220, 38, 38, 0.9)"  # Dark red
        extra_style = "font-weight: 600; border: 2px solid rgba(185, 28, 28, 0.8);"
    elif score > 0.6:
        color = "rgba(239, 68, 68, 0.8)"  # Red
        extra_style = "font-weight: 500; border: 1px solid rgba(220, 38, 38, 0.6);"
    elif score > 0.4:
        color = "rgba(248, 113, 113, 0.7)"  # Light red
        extra_style = "border: 1px solid rgba(239, 68, 68, 0.5);"
    elif score > 0.2:
        color = "rgba(252, 165, 165, 0.6)"  # Very light red
        extra_style = "border: 1px solid rgba(248, 113, 113, 0.4);"
    else:
        color = "rgba(254, 202, 202, 0.4)"  # Barely visible
        extra_style = ""
    
    return color, extra_style

def render_attention(tokens, scores, threshold=0.05):
    """
    Render attention visualization with improved handling of special tokens.
    
    Args:
        tokens: Cleaned tokens (no special tokens)
        scores: Attention scores matching the tokens
        threshold: Minimum score to apply highlighting
    
    Returns:
        HTML string with attention visualization
    """
    if len(tokens) == 0:
        return "<span>No tokens to visualize</span>"
    
    html_parts = []
    
    for token, score in zip(tokens, scores):
        if not token.strip():  # Skip empty tokens
            continue
        
        # Get color and styling
        color, extra_style = get_attention_color(max(score, threshold), enhanced=True)
        
        # Create HTML element
        html_parts.append(
            f'<span class="attention-word" '
            f'style="background-color: {color}; '
            f'padding: 4px 8px; margin: 2px; '
            f'border-radius: 6px; '
            f'display: inline-block; '
            f'transition: all 0.3s ease; '
            f'cursor: pointer; '
            f'{extra_style}" '
            f'title="Token: {token}\nAttention Score: {score:.3f}" '
            f'onmouseover="this.style.transform='translateY(-2px)'; '
            f'this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)';" '
            f'onmouseout="this.style.transform='translateY(0px)'; '
            f'this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)';">'
            f'{token}</span>'
        )
    
    return ' '.join(html_parts)

def show_attention(text, tokenizer, attentions, inputs):
    """
    Main function to show attention visualization with improved special token handling.
    
    This version addresses the issue where [CLS] and [SEP] tokens dominate
    the attention visualization by focusing on content-to-content attention.
    """
    try:
        # Get all tokens
        all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Get content-focused attention scores
        attention_scores = get_content_attention_scores(attentions)
        
        if len(attention_scores) == 0:
            return f"<span>Unable to compute attention for: {text}</span>"
        
        # Clean tokens and match with scores
        clean_tokens, clean_scores = clean_tokens_and_scores(all_tokens, attention_scores)
        
        if len(clean_tokens) == 0:
            return f"<span>No content tokens found in: {text}</span>"
        
        # Render the visualization
        html_output = render_attention(clean_tokens, clean_scores, threshold=0.1)
        
        return html_output
        
    except Exception as e:
        print(f"Error in attention visualization: {e}")
        import traceback
        traceback.print_exc()
        return f"<span style='color: red;'>Error processing attention for: {text}</span>"

def get_attention_stats(attentions):
    """
    Get statistics about attention distribution (excluding special tokens).
    
    Returns:
        Dictionary with attention statistics
    """
    try:
        attention_scores = get_content_attention_scores(attentions)
        if len(attention_scores) == 0:
            return {}
        
        return {
            'mean': float(attention_scores.mean()),
            'std': float(attention_scores.std()),
            'min': float(attention_scores.min()),
            'max': float(attention_scores.max()),
            'num_tokens': len(attention_scores),
            'high_attention_tokens': int((attention_scores > 0.7).sum()),
            'medium_attention_tokens': int(((attention_scores > 0.4) & (attention_scores <= 0.7)).sum()),
            'low_attention_tokens': int((attention_scores <= 0.4).sum())
        }
    except Exception as e:
        print(f"Error computing attention stats: {e}")
        return {}

# Legacy function for backward compatibility
def aggregate_attention(attentions):
    """Legacy function - redirects to new implementation."""
    return get_content_attention_scores(attentions)
