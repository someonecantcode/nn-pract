# Transformers (GPT1)
> https://arxiv.org/pdf/1706.03762

Using weighted aggregation to create interconnected connections between words. Masked for autoregressive behavior.

$ \text{Attention} = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V $

$\text{MultiHead}(Q, K, V) = \text{Attention}(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})$