# Wavenet implementation
> https://arxiv.org/pdf/1609.03499

An autoregressive character-level predictor using wavenet embedding to "convolutionize" parts of context length to generate expected logits of next characters. Instead of the initial implementation of a single embedding layer of dimension N, we slowly crush and divide in multiple iterations.

![embedding portion](wavenet.png)

Used on `names.txt` with the following hyper parameters:

* $lr = 3 \cdot 10^{-3}$
* $n_{embed} =10$

> insert loss graph 

```
insert generated names
```