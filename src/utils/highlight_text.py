import numpy as np
import os
import pdfkit

def highlight_relevant_text(tokens_cleaned, token_importance, save_path, font_prop):
   
    # Normalize importance scores
    importance_range = token_importance.max() - token_importance.min()
    if importance_range == 0:
        importance_scores = np.zeros_like(token_importance)
    else:
        importance_scores = (token_importance - token_importance.min()) / importance_range

    # Create HTML text with highlighted tokens
    html_text = ""
    for tok, score in zip(tokens_cleaned, importance_scores):
        color_intensity = int(255 - score * 255)
        color_hex = f'#ff{color_intensity:02x}{color_intensity:02x}'  # From white to red
        html_text += f'<span style="background-color:{color_hex}; text-decoration: none;">{tok} </span>'

    # Wrap the content in basic HTML structure to prevent external styles from interfering
    html_content = f"""
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
    <p>{html_text}</p>
    </body>
    </html>
    """

    # Save HTML to file
    html_file = os.path.join(save_path, f"highlighted_text.html")
    with open(html_file, "w", encoding='utf-8') as f:
        f.write(html_content)
     
    pdf_file = os.path.join(save_path, f"highlighted_text.pdf")
    pdfkit.from_file(html_file, pdf_file)