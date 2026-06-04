# Transformers (GPT1)
> https://arxiv.org/pdf/1706.03762

Using weighted aggregation to create interconnected connections between words. Masked for autoregressive behavior.

$ \text{Attention} = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V $

$\text{MultiHead}(Q, K, V) = \text{Attention}(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})$

Used on Shakespeare `input.txt` with the following hyper parameters:

* AdamW Optimizer
* $lr = 8 \cdot 10^{-4}$
* $blocksize = 8$
* $n_{embed} = 16$
* $n_{head size} = 40$
* $n_{layer} = 10$

![Loss graph across 1k steps over batches](1kloss.png)