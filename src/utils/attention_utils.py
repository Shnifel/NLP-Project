import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

def display_CLS_layers(attention, tokens, font_prop, seq_len = 30, save_path = "", eng_trans = None):

    #Attention shape is (num_layers, batch_size, num_heads, seq_len, seq_len)
    n_layers = len(attention)
    attention = [attention[l][:,:,0,:seq_len].squeeze(0).squeeze(1).mean(dim = 0).cpu().numpy() for l in range(n_layers)] # batch size will be 1
    attention = np.array(attention)

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, annot=False, cmap='YlOrRd',
                xticklabels=tokens, yticklabels= np.arange(attention.shape[0]))
    plt.xticks(rotation=90, fontsize=8,fontproperties=font_prop)
    plt.yticks(fontsize=8,fontproperties=font_prop)
    plt.xlabel('Tokens' + f' \n(English Translation: \n {eng_trans})' if eng_trans is not None else "",fontproperties=font_prop, fontsize=6)
    plt.ylabel('Layers',fontproperties=font_prop)
    plt.title(f'Attention weight across layers of CLS  \n(Averaged over heads for each layer)', fontproperties= font_prop)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def display_token_to_token(attention, tokens, font_prop, seq_len = 30, save_path = "", eng_trans = None):
    n_layers = len(attention)
    attention = [attention[l][:,:,1:seq_len-1,1:seq_len-1].squeeze(0).mean(dim = 0).cpu().numpy() for l in range(n_layers)] # batch size will be 1
    attention = np.array(attention).mean(axis=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, annot=False, cmap='YlOrRd',
                xticklabels=tokens[1:-1], yticklabels=tokens[1:-1])
    plt.xticks(rotation=90, fontsize=8,fontproperties=font_prop)
    plt.yticks(fontsize=8,fontproperties=font_prop)
    plt.xlabel('Tokens' ,fontproperties=font_prop)
    plt.ylabel('Tokens',fontproperties=font_prop)
    plt.title(f'Attention across network (averaged over heads and layers)', fontproperties= font_prop)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


