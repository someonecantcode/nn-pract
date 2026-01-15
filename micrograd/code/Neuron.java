// has n amount of weights, 1 bias. 
// n is the amount of inputs.
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
        neurons = new Neuron[totaloutput];
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
            output[i] = this.neurons[i].call(input); // basically grab each neuron parameters. no reference, just a copy.
        }
        return output;
    }

    public double[][] parameters() {
        double[][] output = new double[this.neurons.length][this.neurons[0].weights.length]; // copyOf(arraytype[] original, int newLength)
        
        for (int i = 0; i < output.length; i++) {
            output[i] = this.neurons[i].parameters();
        }
        return output;
    }
    @Override
    public String toString() {
        return String.format("Layer of %s", Arrays.deepToString(neurons)); // Layer of [ReLUNeuron(2), ...., ReLUNeuron(2)]
    }
}

class MLP {
    Layer[] layers;

    // input 2 -> numberoutputs [16,16,1]. Layers: [2, 16] -> [16, 16] -> [16, 1] 
    // we count the input layer as layer 0. we say this is a "2 layer mlp"
    public MLP(int numberinputs, int[] numberoutputs) { 
        layers = new Layer[numberoutputs.length + 1];

        // creating [2] [16, 16, 1] -> [2, 16, 16, 1]
        int[] chaining = new int[numberoutputs.length + 1];
        chaining[0] = numberinputs;
        
        for(int i = 0; i < numberoutputs.length; i++){
            chaining[i+1] = numberoutputs[i];
        }
        
        // create layers
        for(int i = 0; i < numberoutputs.length; i++) {
            layers[i] =  new Layer(chaining[i], chaining[i+1], (i != numberoutputs.length - 1));
            /*
            numberoutput = [16, 16, 1]. numberoutput.length = 3
            Layer(2, 16)  ReLU   i = 0
            Layer(16, 16) ReLU   i = 1
            Layer(16, 1)  Linear i = 2  numberoutput.length - 1 is the last layer.
            */
        }
    }

    public void initOptimized(int numberinputs, int[] numberoutputs) {
        layers = new Layer[numberoutputs.length + 1];
         // chaining strat [2, 16] -> [16, 16] -> [16, 1] 
        layers[0] = new Layer(numberinputs, numberoutputs[0]);
        for(int i = 0; i < numberoutputs.length-1; i++){
            layers[i+1] = new Layer(numberoutputs[i], numberoutputs[i+1], (i == numberoutputs.length-2));  // no relu on the last.
        }

        // for(int i = 0; i < numberoutputs.length; i++){
        //     if (i == 0) {
        //         layers[0] = new Layer(numberinputs, numberoutputs[0]);
        //     }
        //     layers[i] = new Layer(numberoutputs[i], numberoutputs[i+1], (i == numberoutputs.length-1));  // no relu on the last.
        // }
    }

}
