import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import nnlib.MLP;
import nnlib.Value;

public class MNIST {

    // hyper parameters
    public static double LEARNING_RATE = 1e-1;
    public static double ALPHA = 1e-1;

    public static long TOTAL_EPOCS = (long) 1e4;
    public static DataLoader loader;

    public static final int OUTPUT_SIZE = 10;

    // so we have inputs, [784 value(0), ... , value(0)] -> mlp -> [10 value(0),...,
    // value(9)] (1entry)
    public static void main(String[] args) throws IOException {
        loader = new DataLoader("./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte");
        // loader.readDataTEST();

        int[] layerparams = { 16, 16, OUTPUT_SIZE };
        MLP m = new MLP(28 * 28, layerparams);

        System.out.println(m.parameters().length);

        trainingLoop(m);
    }

    public static void trainingLoop(MLP m) throws IOException {
        int howMany = 1;
        int counter = 0;
        double loss;

        do {
            m.zeroGrad();
            // get Images + Labels;
            Value[][] expectedLabels = getLabels(howMany);
            Value[][] images = getImages(howMany);
            // System.out.println(Arrays.toString(images[0]));
            // displayValueImage(images[0]);

            // feed forward
            Value[][] preds = new Value[howMany][OUTPUT_SIZE];
            for (int i = 0; i < preds.length; i++) {
                preds[i] = m.call(images[i]);
            }

            // calc loss (L2 + Expected Loss)
            Value reg_loss = getMSEParams(m.parameters()).mult(new Value(ALPHA));
            Value data_loss = getMSELoss(preds, expectedLabels);

            // backwards prop -> grads
            Value total_loss = data_loss.add(reg_loss);
            loss = total_loss.data;
            total_loss.backwards();

            // grad descent
            printInfo(data_loss, reg_loss, preds[0]);
            for (Value param : m.parameters()) {
                param.data += LEARNING_RATE * -param.grad;
            }
            counter++;
            if(counter % 50 == 0) {
                displayValueImage(images[0]);
                System.out.println(Arrays.toString(expectedLabels[0]));
                System.out.println(Arrays.toString(preds[0]));
            }

        } while (counter < 1); //loss >= 1e-1
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
            images[i] = convertIntArraytoValueArray(loader.readImage());
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

    public static Value[] convertIntArraytoValueArray(int[] input) {
        Value[] output = new Value[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = new Value(input[i]);
        }
        return output;
    }

    public static int[] convertValueArraytoIntArray(Value[] input) { // for image displaying
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (int) input[i].data;
        }
        return output;
    }

    public static void displayValueImage(Value[] input) {
        loader.displayImage(convertValueArraytoIntArray(input));
    }

    public static void printInfo(Value data_loss, Value reg_loss, Value[] data) {
        System.out.printf("loss: %.2f | reg_loss: %.3f | data: %s\n", data_loss.data, reg_loss.data,
                Arrays.toString(data));
    }

    public static Value getMSELoss(Value[][] outputs, Value[][] expected) {
        assert (outputs.length == expected.length)
                : "OUTER Predicted output value size not equal to expected value size.";
        assert (outputs[0].length == expected[0].length)
                : "INNER Predicted output value size not equal to expected value size.";

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
