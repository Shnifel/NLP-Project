import numpy as np
import os
import pdfkit
import torch
import re

def html_wrapper(content, font_prop):

    return f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <style>
            span {{
                text-decoration: none;
                font-family: '{font_prop.get_name()}';
            }}
            </style>
        </head>
    <body>
        <p>{content}</p>
    </body>
    </html>
    """


def highlight_relevant_text(tokens, attentions: torch.Tensor, save_path, font_prop, seq_len, original_text, index=0):

    # compute average attention of each input token across all layer/heads/tokens
    attentions = [layer[:,:,:seq_len, :seq_len].squeeze(0).mean(dim=0).mean(dim=1).cpu().numpy()[1:] for layer in attentions]
    token_relevance = np.array(attentions).mean(axis=0)
    relevance_range = np.max(token_relevance) - np.min(token_relevance)
    token_importance = (token_relevance - np.min(token_relevance))/relevance_range if relevance_range != 0 else np.zeros_like(token_relevance)
    token_relevance /= np.sum(token_relevance)
    

    tokens_cleaned = []
    for token in tokens[1:seq_len]:
        if token in ['[SEP]', '[PAD]']:
            tokens_cleaned.append(token)
        elif token.startswith('##'):
            tokens_cleaned.append(token.replace('##', ''))
        else:
            tokens_cleaned.append(token)

    sentences = [[]]
    for token in tokens_cleaned:
        if token == "።" :# full stop token in amharic/tigrinya
            sentences.append([])
        else:
            sentences[-1].append(token)
    original_sentences = re.split(r'(?<=[።])', original_text)

   
    # html text with highlighted individual tokens
    html_text = ""
    for tok, score in zip(tokens_cleaned, token_importance):
        color_intensity = int(255 - score * 255)
        color_hex = f'#{color_intensity:02x}ff{color_intensity:02x}' 
        html_text += f'<span style="background-color:{color_hex}; text-decoration: none;">{tok} </span>'

    html_content = html_wrapper(html_text, font_prop)
    html_file = os.path.join(save_path, f"highlighted_text.html")
    with open(html_file, "w", encoding='utf-8') as f:
        f.write(html_content)
    pdf_file = os.path.join(save_path, f"highlighted_text_{index}_tokens.pdf")
    pdfkit.from_file(html_file, pdf_file)
    os.remove(html_file)

    # html text with sentences highlighted
    html_text = ""
    curr_token = 0
    print(len(tokens_cleaned))
    for i, sentence in enumerate(sentences):
        score = np.sum(token_relevance[curr_token:curr_token+len(sentence)+1]) # sum tokens in sentence
        color_intensity = int(255 - score * 255)
        color_hex = f'#{color_intensity:02x}ff{color_intensity:02x}'
        html_text += f'<span style="background-color:{color_hex}; text-decoration: none;">{original_sentences[i]} </span>'
        curr_token += len(sentence) + 1
        print(score)
    print("DONE")

    html_content = html_wrapper(html_text, font_prop)
    html_file = os.path.join(save_path, f"highlighted_text.html")
    with open(html_file, "w", encoding='utf-8') as f:
        f.write(html_content)
    pdf_file = os.path.join(save_path, f"highlighted_text_{index}_sentences.pdf")
    pdfkit.from_file(html_file, pdf_file)
    os.remove(html_file)