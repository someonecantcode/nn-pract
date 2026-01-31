
import java.util.Arrays;

public class NeuronValue {

    Value[] weights;
    Value bias;
    boolean nonlinear;

    public NeuronValue(int totalinput, boolean nonlinear) {
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

    public NeuronValue(int totalinput) {
        this(totalinput, true);
    }

    // activation
    public Value call(double[] input) {
        assert weights.length == input.length : "input size must the be same as weights size: " + weights.length;

        Value act = new Value(0);
        for (int i = 0; i < input.length; i++) {
            Value xi = new Value(input[i]);
            xi.label = "x" + i;

            Value dot = this.weights[i].mult(xi);
            dot.label = xi.label + this.weights[i].label;
            act = act.add(dot);
        }

        act = act.add(this.bias);
        return this.nonlinear ? act.RELU() : act;
    }

    public Value[] parameters() {
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

// class Layer {
//     Neuron[] neurons;
//     public Layer(int totalinput, int totaloutput, boolean nonlinear) {
//         neurons = new Neuron[totaloutput];
//         for (int i = 0; i < totaloutput; i++) {
//             neurons[i] = new Neuron(totalinput, nonlinear);
//         }
//     }
//     public Layer(int totalinput, int totaloutput) {
//         this(totalinput, totaloutput, true);
//     }
//     public double[] call(double[] input) {
//         double[] output = new double[this.neurons.length];
//         for (int i = 0; i < this.neurons.length; i++) {
//             output[i] = this.neurons[i].call(input);
//         }
//         return output;
//     }
//     public double[][] parameters() {
//         double[][] output = new double[this.neurons.length][this.neurons[0].weights.length];
//         for (int i = 0; i < output.length; i++) {
//             output[i] = this.neurons[i].parameters();
//         }
//         return output;
//     }
//     @Override
//     public String toString() {
//         return String.format("Layer of %s", Arrays.deepToString(neurons));
//     }
// }
// public class MLPValue {
//     Layer[] layers;
//     public MLPValue(int numberinputs, int[] numberoutputs) { 
//         this.layers = new Layer[numberoutputs.length];
//         int[] chaining = new int[numberoutputs.length + 1];
//         chaining[0] = numberinputs;
//         for(int i = 0; i < numberoutputs.length; i++){
//             chaining[i+1] = numberoutputs[i];
//         }
//         for(int i = 0; i < numberoutputs.length; i++) { 
//             this.layers[i] =  new Layer(chaining[i], chaining[i+1], (i != numberoutputs.length - 1));
//         }
//     }
//     public void initOptimized(int numberinputs, int[] numberoutputs) {
//         layers = new Layer[numberoutputs.length + 1];
//          // chaining strat [2, 16] -> [16, 16] -> [16, 1] 
//          // [2, 16, 16, 1]
//         layers[0] = new Layer(numberinputs, numberoutputs[0]);
//         for(int i = 1; i < numberoutputs.length+1; i++){
//             layers[i] = new Layer(numberoutputs[i], numberoutputs[i+1], (i == numberoutputs.length));  // no relu on the last index.
//         }
//         // for(int i = 0; i < numberoutputs.length+1; i++){
//         //     if (i == 0) {
//         //         layers[0] = new Layer(numberinputs, numberoutputs[0]);
//         //     }
//         //     layers[i] = new Layer(numberoutputs[i], numberoutputs[i+1], (i == numberoutputs.length-1));  // no relu on the last.
//         // }
//     }
//     public double[] call(double[] input) {
//         double[] output = Arrays.copyOf(input, input.length); 
//         // double[] output = new output[input.length] System.arraycopy(input, 0, output, 0, input.length)
//         // forward pass -> output of the last layer
//         for (int i = 0; i < this.layers.length; i++) {
//             output = this.layers[i].call(output);
//         }
//         return output; 
//     }
//     public double[][][] parameters() {
//         double[][][] output = new double[this.layers.length][][];
//         for(int layerindex = 0; layerindex < output.length; layerindex++) {
//             output[layerindex] = this.layers[layerindex].parameters();            
//         }
//         return output;
//     }
//     @Override
//     public String toString(){
//         return String.format("MLP of %s", Arrays.deepToString(layers));
//     }
// }
