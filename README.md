# Technical-task-AIDF
Pre-project Task - Hallucination Detection

All files run on a NVIDIA A100 GPU rented from vest.ai with model LlaMa-3.1-8B requested from Meta official website.
### Repository Structure
```
Technical-task-AIDF/
│
├── baseline/                                 
│   ├── baseline.py     
│   ├── utils.py                              
│   └── __init__.py
│
├── HaMI/                                     
│   ├── hami.py                               
│   ├── enhanced_hami.py                     
│   └── __init__.py
│
├── src/                                      
│   ├── api.py                                
│   ├── data.py                               
│   ├── final_select.py                       
│   ├── model.py                             
│   ├── multi_sample.py                      
│   ├── process_data.py                       
│   ├── prompt.py                             
│   ├── random_pairs.py                       
│   ├── refined_set.py                        
│   └── __init__.py
│
├── main.py                                   
├── requirements.txt                         
├── LICENSE                                   
├── .gitignore                               
└── README.md                                 
```
### Detailed process as follows:
Clone the repository to Google Colab
```{bash}
git clone https://github.com/jiaozijun666/Technical-task-AIDF.git
cd Technical-task-AIDF
```
Creat virtual environment
```{python}
python3 -m venv .venv
source .venv/bin/activate
```
Download the model Llama-3.1-8B to environment
```{bash}
# Download model from official website
pip install llama-stack
llama model download --source meta --model-id  Llama3.1-8B

# Check status
python - <<'PY'
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
p = os.environ["MULTI_MODEL_DIR"]
tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
mdl = AutoModelForCausalLM.from_pretrained(p, local_files_only=True)
print("ok:", type(tok).__name__, type(mdl).__name__)
PY

# (Optional, if fail to download the model, then) Download ModelScope
source /workspace/Technical-tesk-AIDF/.venv/bin/active 2>/dev/null || true
pip install modelscope transoformers accelerate bitsandbytes sentencepiece

# (Optional) Transformer mode
python - <<'PY'
from modelscope import snapshot_download
d = snapshot_download(
    'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    cache_dir='/workspace/models',
    revision=None
)
print(d)
PY

# Check status
ls -lh /workspace/models/LLM-Research/Meta-Llama-3.1-8B-Instruct | head
du -sh /worksoace/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
```
Introducing openai api key
```{bash}
pip install --upgrade openai
export OPENAI_API_KEY="sk-xxxxxxxx"
```

Downloading the necessary packages
```{bash}
pip install -r requirements.txt
```

Running the piplines in following order
```{bash}
# Ensure runs on the instruced model
ln -sfn /workspace/models/LLM-Research/Meta-Llama-3___1-8B-Instruct \
        /workspace/models/LLM-Research/Meta-Llama-3.1-8B-Instruct #When instruct into local evironment, name of the model file may vary
export MULTI_MODEL_DIR=/workspace/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
export MULTI_MODEL_DIR=/workspace/models/LLM-Research/Meta-Llama-3.1-8B-Instruct

# Setseed 
export SEED=42

# (Optional due to the limited hashrate) Running on a small number of data to test
MULTI_LIMIT=50 python main.py
#You can set the number as you like, if you just run python main.py, it will takes longer time because the multi_sample.py process will generate 2000 pairs of data training 5 times. Reduce the number to reduce the running time, but would lead to uncertainty to the result.

# (If you have enough hardware and hashrate) Running the whole dataset 
python main.py
```
