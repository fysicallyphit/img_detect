from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def extract_qkv():
    Q_matrices = []
    K_matrices = []
    V_matrices = []

    def hook_Q(module, input, output):
        Q_matrices.append(output) 

    def hook_K(module, input, output):
        K_matrices.append(output)

    def hook_V(module, input, output):
        V_matrices.append(output)  
            
    model.encoder.layer[0].attention.self.query.register_forward_hook(hook_Q)
    model.encoder.layer[0].attention.self.key.register_forward_hook(hook_K)
    model.encoder.layer[0].attention.self.value.register_forward_hook(hook_V)

    inputs = tokenizer("the quick brown fox jumped over the lazy dog" * 20,  return_tensors ="pt")
    K_matrices.clear() 
    V_matrices.clear()
    Q_matrices.clear() 
    outputs = model(**inputs)
    seq_len = inputs["input_ids"].shape[1]

    Q = Q_matrices[0]
    Q = Q.reshape(1, seq_len, 12, 64).transpose(1,2)
    Q_head = Q[0,0,:,:].detach().numpy()

    K = K_matrices[0]
    K = K.reshape(1, seq_len, 12, 64).transpose(1,2)
    K_head = K[0,0,:,:].detach().numpy()

    V = V_matrices[0]
    V = V.reshape(1, seq_len, 12, 64).transpose(1,2)
    V_head = V[0,0,:,:].detach().numpy()

    return outputs, Q_head, K_head, V_head
