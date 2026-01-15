// has n amount of weights, 1 bias. 
// n is the amount of inputs.

import java.lang.reflect.Array;
import java.util.Arrays;

public class Neuron {

    double[] weights;
    double bias;
    boolean nonlinear;

    public Neuron(int totalinput, boolean nonlinear) {
        weights = new double[totalinput];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random(); // fill weights with random double from 0.0 to 1.0
        }
        this.bias = 0;
        this.nonlinear = nonlinear;
    }

    public Neuron(int totalinput) { // for optional nonlinear parameter
        this(totalinput, true); // call neuron constructor
    }

    //activation
    public double call(double[] input) {
        assert weights.length == input.length : "input size must the be same as weights size: " + weights.length;

        double act = 0;
        for (int i = 0; i < input.length; i++) {
            act += this.weights[0] * input[0]; // dot product weights @ input.
        }

        act += this.bias;
        return this.nonlinear ? Math.max(0, act) : act; //relu activation function. only take positive
    }

    public double[] parameters() {
        double[] output = Arrays.copyOf(this.weights, this.weights.length + 1); // copyOf(arraytype[] original, int newLength)
        output[this.weights.length] = this.bias; // add bias as the last index

        return output;
    }

    @Override
    public String toString() {
        String type = this.nonlinear ? "ReLU" : "Linear";
        return String.format("%sNeuron(%d)", type, this.weights.length); // ReLUNeuron(number of inputs)
    }
}

class Layer {

    Neuron[] neurons;

    public Layer(int totalinput, int totaloutput, boolean nonlinear) { // later use varargs for more hyperparams
        for (int i = 0; i < totaloutput; i++) {
            neurons[i] = new Neuron(totalinput, nonlinear);
        }
    }

    public Layer(int totalinput, int totaloutput) {
        this(totalinput, totaloutput, true);
    }

    public double[] call(double[] input) {

        double[] output = new double[this.neurons.length];

        for (int i = 0; i < this.neurons.length; i++) {
            output[i] = this.neurons[i].call(input); // basically grab each neuron output
        }
        return output;
    }

    public double[][] parameters() {
        double[][] output = new double[this.neurons.length][this.neurons[0].weights.length]; // copyOf(arraytype[] original, int newLength)
        
        for (int i = 0; i < output.length; i++) {
            output[i] = this.neurons[i];
        }
        return output;
    }
    // @Override
    // public String toString() {
    //     String type = this.nonlinear ? "ReLU" : "Linear";
    //     return String.format("%sNeuron(%d)", type, this.weights.length); // ReLUNeuron(number of inputs)
    // }
}

class MLP {

}
