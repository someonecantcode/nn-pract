import java.io.IOException;
import nnlib.MLP;
import nnlib.Value;

public class MNIST {

    // hyper parameters
    public static double LEARNING_RATE = 1e-1;
    public static double ALPHA = 1e-2;

    public static long TOTAL_EPOCS = (long) 1e4;
    public static DataLoader loader;

    // so we have inputs, [784 value(0), ... , value(0)] -> mlp -> [10 value(0), ..., value(9)] (1entry)
    public static void main(String[] args) throws IOException {
        loader = new DataLoader("./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte");
        loader.readDataTEST();

        int[] layerparams = {16, 16, 3};
        MLP m = new MLP(28 * 28, layerparams);

        //  System.out.println(m.parameters().length);
    }

    // public static void trainingLoop(MLP m) {
    //     Value[] expected = getLabels();
    //     do {
    //         m.zeroGrad();

    //         // calc loss (L2 + Expected Loss)
    //         Value reg_loss;
    //         Value data_loss;

    //         // backwards prop -> grads
    //         printInfo(data_loss, reg_loss);
    //         // grad descent
    //         for (Value param : m.parameters()) {
    //             param.data += LEARNING_RATE * -param.grad;
    //         }
    //     } while (getMSEloss(outputs, expected));
    // }

    public static Value[] getLabels(int howMany) throws IOException {
        Value[] labels = new Value[howMany];
        for (int i = 0; i < labels.length; i++) {
            labels[i].data = loader.readLabel();
        }
        return labels;
    }

    public static Value[][] getImages(int howMany) throws IOException {
        Value[][] images = new Value[howMany][784];
        for (int i = 0; i < images.length; i++) {
            images[i] = convertIntArraytoValueArray(loader.readImage());
        }
        return images;
    }

    public static Value[] convertIntArraytoValueArray(int[] input) {
        Value[] output = new Value[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i].data = input[i];
        }
        return output;
    }

    public static void printInfo(Value data_loss, Value reg_loss) {
        System.out.printf("\nloss: %.2f | reg_loss: %.2f | data: %s ", data_loss.data, reg_loss.data);
    }

    public static Value getMSEloss(Value[][] outputs, Value[] expected) {
        Value acc = new Value(0);

        for (Value[] output : outputs) {
            for (int j = 0; j < expected.length; j++) { // expected.length incase we only care about certain entries
                // (expected[i] - output[i])^2
                Value calc = (expected[j].sub(output[j])).pow(2);
                acc = acc.add(calc);
            }
        }
        return acc.div(new Value(outputs.length * outputs[0].length));
    }

}
