from src.extract import extract_qkv
from src.compress import compute_svd
from src.evaluate import compression_experiment
from src.plot import plot_results
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

outputs, Q_head, K_head, V_head = extract_qkv()
Uk, Sk, Vtk, Uv, Sv, Vtv = compute_svd(K_head, V_head)
r_range, attention_errors, compressed_KV_memory, full_KV_memory = compression_experiment(Q_head, K_head, V_head, Uk, Sk, Vtk, Uv, Sv, Vtv)
plot_results(r_range, attention_errors, compressed_KV_memory, full_KV_memory)