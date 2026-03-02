
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import nnlib.MLP;
import nnlib.Value;

public class MNIST {

    // hyper parameters
    public static double LEARNING_RATE = 5e-2;
    public static double ALPHA = 1e-3;

    public static long TOTAL_EPOCS = (long) 1e1;
    public static DataLoader loader;

    public static final int OUTPUT_SIZE = 10;

    // so we have inputs, [784 value(0), ... , value(0)] -> mlp -> [10 value(0),...,
    // value(9)] (1entry)
    public static void main(String[] args) throws IOException {
        loader = new DataLoader("./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte");
        // loader.readDataTEST();

        int[] layerparams = {32, 32, OUTPUT_SIZE};
        MLP m = new MLP(28 * 28, layerparams);

        System.out.println(m.parameters().length);

        trainingLoop(m);
    }

    public static void trainingLoop(MLP m) throws IOException {
        int howMany = 10;
        int counter = 0;
        double loss;

        do {
            m.zeroGrad();
            // get Images + Labels;
            Value[][] expectedLabels = getLabels(howMany);
            Value[][] images = getImages(howMany);
            
            // System.out.println(Arrays.toString(images[0]));
            // displayValueImage(images[0]);
            // feed forward + softmax
            Value[][] preds = new Value[howMany][OUTPUT_SIZE];
            for (int i = 0; i < preds.length; i++) {
                preds[i] = softmax(m.call(images[i]));
            }

            // calc loss (L2 + Expected Loss)
            Value reg_loss = getMSEParams(m.parameters()).mult(new Value(ALPHA));
            Value data_loss = getCrossEntropyLoss(preds, expectedLabels);
            // backwards prop -> grads
            Value total_loss = data_loss.add(reg_loss);
            loss = total_loss.data;
            total_loss.backwards();

            // // grad descent
            for (Value param : m.parameters()) {
                param.data += LEARNING_RATE * -param.grad;
            }
            counter++;

            if (counter % TOTAL_EPOCS == 0) {
                displayValueImage(images[0]);
                System.out.println(Arrays.toString(expectedLabels[0]));
                System.out.println(Arrays.toString(preds[0]));
                System.out.println();
                printInfo(data_loss, reg_loss, preds, expectedLabels);
            }
        } while (counter < 1000); //loss >= 1e-5 
    }

    public static Value[][] getLabels(int howMany) throws IOException {
        Value[] labels = new Value[howMany];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = new Value(loader.readLabel());

        }
        return packageExpectedLabels(labels);
    }

    public static Value[][] getImages(int howMany) throws IOException {
        Value[][] images = new Value[howMany][784];
        for (int i = 0; i < images.length; i++) {
            images[i] = convertDoubleArraytoValueArray(loader.readImage());
        }
        return images;
    }

    public static Value[][] packageExpectedLabels(Value[] labels) {
        Value[][] output = new Value[labels.length][OUTPUT_SIZE];
        for (int i = 0; i < output.length; i++) {
            // create 10 (0~9) length vector expected value [1,0,0,0,0,0,0,0]
            for (int j = 0; j < output[i].length; j++) {
                output[i][j] = new Value((j == labels[i].data) ? 1 : 0);
            }
        }
        return output;
    }

    public static Value[] convertDoubleArraytoValueArray(double[] input) {
        Value[] output = new Value[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = new Value(input[i]);
        }
        return output;
    }

    public static double[] convertValueArraytoDoubleArray(Value[] input) { // for image displaying
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i].data;
        }
        return output;
    }

    public static void displayValueImage(Value[] input) {
        loader.displayImage(convertValueArraytoDoubleArray(input));
    }

    public static void printInfo(Value data_loss, Value reg_loss, Value[][] data, Value[][] expected) {
        for (int i = 0; i < data.length; i++) {
            System.out.printf("loss: %.3f | reg_loss: %.3f | data: %s | data: %s | expected: %s\n", data_loss.data, reg_loss.data,
                    Arrays.toString(data[i]), getMaxValueIndex(data[i]), getMaxValueIndex(expected[i]));
        }

    }

    public static int getMaxValueIndex(Value[] input) {
        int index = 0;
        for (int i = 0; i < input.length; i++) {
            if (input[i].data > input[index].data) {
                index = i;
            }
        }
        return index;
    }

    public static Value[] softmax(Value[] input) {
        // find max number
        Value max = new Value(input[getMaxValueIndex(input)].data);
        Value[] softmax_output = new Value[input.length];
        // first sum e^z
        List<Value> input_exp = new ArrayList<>();
        for (int i = 0; i < input.length; i++) {
            softmax_output[i] = (input[i].sub(max)).exp();
            input_exp.add(softmax_output[i]);
        }
        Value sum = Value.sum(input_exp);

        // e^z / sum
        for (int i = 0; i < softmax_output.length; i++) {
            softmax_output[i] = softmax_output[i].div(sum);
        }

        return softmax_output;
    }

    public static Value getCrossEntropyLoss(Value[][] softmax_outputs, Value[][] expected) {
        final double EPSILON = 1e-8;

        List<Value> losses = new ArrayList<>();
        Value loss_output;

        for (int i = 0; i < softmax_outputs.length; i++) {
            for (int j = 0; j < softmax_outputs[i].length; j++) {
                // expected * log(softmax(pred))
                Value ln_softmax = (softmax_outputs[i][j].add(new Value(EPSILON))).ln();
                Value calc = expected[i][j].mult(ln_softmax);
                losses.add(calc);
            }
        }

        loss_output = Value.sum(losses);
        loss_output = loss_output.div(new Value(softmax_outputs.length));
        return loss_output.neg();
    }

    public static Value getMSELoss(Value[][] outputs, Value[][] expected) {
        assert (outputs.length == expected.length) : "OUTER Predicted output value size not equal to expected value size.";
        assert (outputs[0].length == expected[0].length) : "INNER Predicted output value size not equal to expected value size.";

        List<Value> losses = new ArrayList<>();
        Value loss_output;
        // output pred 1 [0, 1, 0.2, 0.3, 0.04, 0.05, 0.06, 0.07, 0.08, 0.009]
        // expected 1 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        for (int i = 0; i < outputs.length; i++) {
            for (int j = 0; j < outputs[i].length; j++) {
                // (expected[][] - output[][])^2
                Value calc = (expected[i][j].sub(outputs[i][j])).pow(2);
                losses.add(calc);
            }
        }

        loss_output = Value.sum(losses);
        loss_output = loss_output.div(new Value(outputs.length * outputs[0].length));

        return loss_output;
    }

    public static Value getMSEParams(Value[] params) {
        List<Value> params_squared = new ArrayList<>();
        Value loss_output;
        for (Value p : params) {
            Value calc = p.pow(2);
            params_squared.add(calc);
        }

        loss_output = Value.sum(params_squared);
        loss_output = loss_output.div(new Value(params.length));
        return loss_output;
    }
}
