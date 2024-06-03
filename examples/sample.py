# Copyright (C) Ronsor Labs. All rights reserved.
#
# The license of this software is specified in the LICENSE file at the root of
# this repository.

import sys, torch

sys.path.append('./')
from rwkv_simple import RWKV, WKVBackend, wkv6_kernel, WorldTokenizer

# Options

model_path = 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
device = 'cuda'

vocab_path = None # 'rwkv_vocab_v20230424.txt'

prompt = "User: What is 2 + 2\nAssistant:"
sampler = 'dart' # 'topk' or 'dart'
max_tokens = 50
temperature = 1.0
top_p = 0.5
top_k = 5

# - Backends for WKV6 kernel (i.e. our attention replacement)
backends = [WKVBackend.FLA, WKVBackend.CUDA, WKVBackend.PYTORCH_OPTIMIZED]

# Load model

ckpt = torch.load(model_path, map_location=device)

vocab_size, d_model = ckpt['emb.weight'].shape
_, d_ffn = ckpt['blocks.0.ffn.value.weight'].shape
n_layer = int(list(filter(lambda x: x.startswith('blocks.'), ckpt.keys()))[-1].split('.')[1]) + 1

print(f"Model: d_model={d_model}, vocab_size={vocab_size}, d_ffn={d_ffn}, n_layer={n_layer}")

model = RWKV(
  d_model=d_model,
  expand=d_ffn/d_model,
  n_layer=n_layer,
  vocab_size=vocab_size,
  device='meta', # We do not want to initialize any tensors, since we're loading an existing checkpoint
  dtype=torch.bfloat16,
)

model.load_state_dict(ckpt, assign=True) # since we use device='meta', we have to assign the new tensors, rather than copy
model.eval()

def run_model(*args, **kwargs):
  global backends
  with wkv6_kernel(backends), torch.no_grad():
    return model(*args, **kwargs)

# Tokenize

tokenizer = WorldTokenizer(vocab_path)

prompt_tokens = tokenizer.encode(prompt)

# Process prompt

prompt_tensor = torch.tensor([prompt_tokens]).to(device)

y, state = run_model(prompt_tensor, state=None, need_state=True)

print('Prompt:\n----------')
print(prompt)
print('\nTokenized:')
print(prompt_tokens)

# Sample

def dart_sample(out, temperature=1.0, top_p=0.5):
  out = torch.softmax(out, dim=-1)

  dart = torch.rand(1, device=out.device)
  dart = dart.pow(temperature)
  dart = top_p * dart + (1 - top_p) * 1.0
  dart = out.min() * (1 - dart) + out.max() * dart

  out = torch.argmin(torch.abs(out - dart)).item()
  return out

def topk_sample(out, temperature=1.0, top_k=None):
  out = out / temperature
  if top_k is not None:
    v, _ = torch.topk(out, min(top_k, out.size(-1)))
    out[out < v[:, [-1]]] = -float('inf')

  out = torch.softmax(out, dim=-1)
  return torch.multinomial(out, num_samples=1).item()

print('\nOutput:\n----------')

torch.manual_seed(42)

n = 1
while n < max_tokens:
  if sampler == 'dart':
    token = dart_sample(y[:, -1, :], temperature, top_p)
  elif sampler == 'topk':
    token = topk_sample(y[:, -1, :], temperature, top_k)
  else:
    assert False, "sampler must be 'dart' or 'topk'"

  if token == 0:
    break

  n += 1
  print(tokenizer.decode([token]), end='')
  sys.stdout.flush()

  y, state = run_model(torch.tensor([[token]]).to(device), state=state, need_state=True)

print('\n')
