'''
	Contoh Deloyment untuk Domain Natural Language Processing (NLP)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify

import pandas as pd
import text_hammer as th
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
from transformers import TFBertModel
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import re
import warnings
warnings.filterwarnings('ignore')

PRE_TRAINED_MODEL = 'indobenchmark/indobert-large-p2'
max_len = 50

bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
loaded_model = tf.keras.models.load_model('modelbase.h5', custom_objects={'TFBertModel':TFBertModel})

abrv = pd.read_csv('abrv.csv')

def normalize_abrv(x):
    words = x.split(' ')
    attrfix = abrv[' fix'].tolist()
    attralias = abrv['alias'].tolist()
    normalized = []
    for w in words:
        found = False
        for i, a in enumerate(attralias):
            if w == a: 
                normalized.append(attrfix[i])
                found = True
        if not found: normalized.append(w) 
    return " ".join(normalized)

def analyze(x):
    x = str(x).lower()
    x = th.cont_exp(x)
    x = th.remove_special_chars(x)
    x = th.remove_accented_chars(x)
    x = re.sub(r'[-+]?[0-9]+', '', x)
    x = normalize_abrv(x)

    x_val = bert_tokenizer(
        text=x,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True) 
        
    validation = loaded_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    return validation

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

stopwords_ind = None
key_norm      = None
factory       = None
stemmer       = None
vocab         = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]		
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Nilai default untuk string input 
	text_input = ""
	
	if request.method=='POST':
		# Set nilai string input dari pengguna
		text_input = request.form['data']
		out = analyze(text_input)


		if(out[0][0] > out[0][1]):
			hasil_prediksi = "BULLY"
		else:
			hasil_prediksi = "NORMAL"
		
		# Return hasil prediksi dengan format JSON
		return jsonify({
			"data": hasil_prediksi,
		})

# =[Main]========================================

app.run(host="localhost", port=5000, debug=True)
	
	


