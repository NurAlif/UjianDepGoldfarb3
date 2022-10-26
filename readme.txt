===================
[Packages List]
===================
Python   3.8.5
Flask    2.1.0
Werkzeug 2.1.2
joblib   1.0.0
pandas   1.4.2
numpy    1.21.6
sklearn  0.24.1
nltk     3.7
Sastrawi 1.0.1

=======================================
[Environment & Packages Installation]
=======================================

conda create -n nlp_app_1 python=3.8.5
activate nlp_app_1

pip install Flask==2.1.0
pip install Werkzeug==2.1.2
pip install joblib==1.0.0
pip install pandas==1.4.2
pip install numpy==1.21.6
pip install scikit-learn==0.24.1
pip install nltk==3.7
pip install Sastrawi==1.0.1

python
import nltk
nltk.download('stopwords')
