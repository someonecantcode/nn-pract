# micrograd


<p align="center">
  <img src="./Images/thumbnail.png" alt="intro image" width="300">
</p>


* Ideas on backpropogation chain rule to obtain partial derivatives
* Using individual partial derivatives (gradient) to minimize loss by shifting biases and weights.
* Teaches us Neuron, Layer, MLP archetictures.

## Backpropogation

$$
\begin{align}

& \frac{dz}{dx} = \frac{dz}{dy}  \frac{dy}{dx}  \\
& \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{u}}\frac{\partial \mathbf{u}}{\partial \mathbf{x}}

\end{align}
$$

## Gradient

$$
\begin{align}
& \nabla f(x) = \ \langle f_x, f_y , f_z , \dots, f_n\rangle \\
& w := w - \eta \nabla f(x)  \\
\end{align}
$$

Why negative? \
We want to minimize our loss, not maximize it.

## Neuron, Layer, MLP

* Singular Neuron summations
$$ z = b +\sum_{i=1}^{n} w_i x_i  $$

* Activation Sigmoid
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

* GELU

$$ \begin{align} & \mathrm{GELU}(x) = x \, \Phi(x)  =  \frac{x}{2}\left[1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right] \\
& \mathrm{GELU}(x)  \approx 0.5x \left(1 + \tanh\!\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^{3}\right)\right]\right)

\end{align}$$
