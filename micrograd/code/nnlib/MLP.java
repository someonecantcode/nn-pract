package micrograd.nnlib;

import java.util.Arrays;

abstract class Module {

    public abstract Value[] parameters(); // need squash all parameters into a 1D array

    public void zeroGrad() {
        for (Value p : parameters()) {
            p.grad = 0;
        }
    }
}

class Neuron extends Module {

    Value[] weights;
    Value bias;
    boolean nonlinear;

    public Neuron(int totalinput, boolean nonlinear) {
        weights = new Value[totalinput];
        for (int i = 0; i < totalinput; i++) {
            weights[i] = new Value((Math.random() * 2) - 1);
            weights[i].label = "w" + i;
            // optimized weights[i] = ThreadLocalRandom.current().nextDouble(-1.0, 1.0);
        }
        this.bias = new Value(0);
        this.bias.label = "b";
        this.nonlinear = nonlinear;
    }

    public Neuron(int totalinput) {
        this(totalinput, true);
    }

    // activation
    public Value call(Value[] input) {
        assert weights.length == input.length : "input size must the be same as weights size: " + weights.length;

        Value act = new Value(0);
        for (int i = 0; i < input.length; i++) {
            Value dot = this.weights[i].mult(input[i]);
            act = act.add(dot);
        }

        act = act.add(this.bias);
        act.label = "act_final";
        return this.nonlinear ? act.RELU() : act; // RELU = 0 causes all gradients to be 0.
    }

    @Override
    public Value[] parameters() { // creates references allowing for mutation of values
        Value[] output = Arrays.copyOf(this.weights, this.weights.length + 1);
        output[this.weights.length] = this.bias;

        return output;
    }

    @Override
    public String toString() {
        String type = this.nonlinear ? "ReLU" : "Linear";
        return String.format("%sNeuron(%d)", type, this.weights.length);
    }
}

class Layer extends Module {

    Neuron[] neurons;

    public Layer(int totalinput, int totaloutput, boolean nonlinear) {
        neurons = new Neuron[totaloutput];
        for (int i = 0; i < totaloutput; i++) {
            neurons[i] = new Neuron(totalinput, nonlinear);
        }
    }

    public Layer(int totalinput, int totaloutput) {
        this(totalinput, totaloutput, true);
    }

    public Value[] call(Value[] input) {
        Value[] output = new Value[this.neurons.length];
        for (int i = 0; i < this.neurons.length; i++) {
            output[i] = this.neurons[i].call(input);
        }
        return output;
    }

    @Override
    public Value[] parameters() {
        //calc total params in layer: weights/bias of each neuron * total neurons
        int total_params = this.neurons.length * this.neurons[0].parameters().length;
        Value[] output = new Value[total_params];

        int offset = 0;
        for (Neuron n : this.neurons) {
            System.arraycopy(n.parameters(), 0, output, offset, n.parameters().length);
            offset += n.parameters().length;
        }

        return output;
    }

    @Override
    public String toString() {
        return String.format("Layer of %s", Arrays.deepToString(neurons));
    }
}

public class MLP extends Module {
    Layer[] layers;
    public MLP(int numberinputs, int[] numberoutputs) { 
        this.layers = new Layer[numberoutputs.length];
        int[] chaining = new int[numberoutputs.length + 1];
        chaining[0] = numberinputs;
        
        System.arraycopy(numberoutputs, 0, chaining, 1, numberoutputs.length);
        for(int i = 0; i < numberoutputs.length; i++) { 
            this.layers[i] =  new Layer(chaining[i], chaining[i+1], (i != numberoutputs.length - 1)); //last one is activation RELU
        }
    }

    public Value[] call(Value[] input) {
        // forward pass -> output of the last layer

        Value[] output = this.layers[0].call(input);
        for (int i = 1; i < this.layers.length; i++) {
            output = this.layers[i].call(output);
        }
        return output; 
    }

    @Override
    public Value[] parameters() {
        int total_params = 0;
        for(Layer l : this.layers){
           total_params += l.neurons.length * l.neurons[0].parameters().length;
        }

        Value[] output = new Value[total_params];

        int offset = 0;
        for (Layer l : this.layers) {
            System.arraycopy(l.parameters(), 0, output, offset, l.parameters().length);
            offset += l.parameters().length;
        }
        return output;
    }

    @Override
    public String toString(){
        return String.format("MLP of %s", Arrays.deepToString(layers));
    }
}
