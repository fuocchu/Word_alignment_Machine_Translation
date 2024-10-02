import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import base64
import io
import pickle


app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')


def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))

def group_subwords(tokens):
    grouped_tokens = []
    current_token = ""
    for token in tokens:
        if token.startswith("##"):
            current_token += token[2:]
        else:
            if current_token:
                grouped_tokens.append(current_token)
            current_token = token
    if current_token:
        grouped_tokens.append(current_token)
    return grouped_tokens

def align_sentences(sentence_vn, sentence_en):
    tokens_vn = tokenizer(sentence_vn, return_tensors='pt')
    tokens_en = tokenizer(sentence_en, return_tensors='pt')
    
    with torch.no_grad():
        embeddings_vn = model(**tokens_vn)['last_hidden_state']
        embeddings_en = model(**tokens_en)['last_hidden_state']
        
    tokens_vn_text = group_subwords(tokenizer.convert_ids_to_tokens(tokens_vn['input_ids'][0]))
    tokens_en_text = group_subwords(tokenizer.convert_ids_to_tokens(tokens_en['input_ids'][0]))
    

    tokens_vn_text = [token for token in tokens_vn_text if token not in ['[CLS]', '[SEP]']]
    tokens_en_text = [token for token in tokens_en_text if token not in ['[CLS]', '[SEP]']]
    
    alignment = []
    
    for i, vec_vn in enumerate(embeddings_vn[0]):
        if i == 0 or i == len(embeddings_vn[0]) - 1:
            continue
        best_match = -1
        best_score = float('-inf')
        for j, vec_en in enumerate(embeddings_en[0]):
            if j == 0 or j == len(embeddings_en[0]) - 1:
                continue
            score = cosine_similarity(vec_vn, vec_en).item()
            if score > best_score:
                best_score = score
                best_match = j
        
        alignment.append((i - 1, best_match - 1, best_score))
            
    for align in alignment:
        vn_idx, en_idx, score = align
        if vn_idx < len(tokens_vn_text) and en_idx < len(tokens_en_text):
            print(f"Vietnamese word '{tokens_vn_text[vn_idx]}' align with English word '{tokens_en_text[en_idx]}' (score: {score:.4f})")
    
    return tokens_vn_text, tokens_en_text, alignment


def plot_align(tokens_vn, tokens_en, alignment):
    fig, ax = plt.subplots(figsize=(12, 6))

    vn_offset = 0.5
    en_offset = 1.5

    for i, token in enumerate(tokens_vn):
        ax.text(vn_offset, len(tokens_vn) - i, token, fontsize=12, verticalalignment='center', ha='right')

    for i, token in enumerate(tokens_en):
        ax.text(en_offset, len(tokens_en) - i, token, fontsize=12, verticalalignment='center', ha='left')

    for vn_idx, en_idx, _ in alignment:
        ax.plot([vn_offset, en_offset], [len(tokens_vn) - vn_idx, len(tokens_en) - en_idx], 'k-', lw=1)

    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, max(len(tokens_vn), len(tokens_en))+1)
    ax.axis('off')
    
    img =io.BytesIO()
    plt.savefig(img,format='png')
    img.seek(0)
    img_base64=base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64
#save model
pickle.dump(model, open('model.pkl', 'wb'))

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/align', methods=['POST'])

def align():
    data = request.json
    sentence_vn = data['sentence_vn']
    sentence_en = data['sentence_en']
    tokens_vn, tokens_en, alignment = align_sentences(sentence_vn, sentence_en)
    img_base64 = plot_align(tokens_vn, tokens_en, alignment)
    
    response = {
        'alignment': [
    (tokens_vn[vn_idx], tokens_en[en_idx]) 
    for vn_idx, en_idx, _ in alignment 
    if vn_idx < len(tokens_vn) and en_idx < len(tokens_en)
],
        'image': img_base64
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


