# rwkv_simple

`rwkv_simple` is an easy-to-use implementation of RWKV-6 (x060). We support
multiple WKV kernels (a Triton-based one from Flash Linear Attention, the official
CUDA kernel, and a pure-PyTorch one).

## Flash Linear Attention

We use the WKV6 kernel from the Flash Linear Attention (FLA) project by default,
so for best performance, please install FLA directly from its repository:

```
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```

See: <https://github.com/sustcsonglin/flash-linear-attention>.

## License

Copyright © 2024 Ronsor Labs. Licensed under the Apache License, version 2.0.
