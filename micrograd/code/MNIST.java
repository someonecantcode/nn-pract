
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import nnlib.*;

public class MNIST {

    // hyper parameters
    public static final double LEARNING_RATE = 3e-3;
    public static final double ALPHA = 1e-2;
    public static final double EPSILON = 1e-8;

    // for adam :)
    public static final double BETA_1 = 0.9;
    public static final double BETA_2 = 0.999;
    public static double[] m;
    public static double[] v;

    public static long TOTAL_EPOCS = (long) 1e2;
    public static final int OUTPUT_SIZE = 10;
    public static DataLoader loader;
    public static ModelSaver saver;

    // so we have inputs, [784 value(0), ... , value(0)] -> mlp -> [10 value(0),...,
    // value(9)] (1entry)
    public static ArrayList<Double> acc = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        loader = new DataLoader("./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte");
        saver = new ModelSaver();
        // loader.readDataTEST();

        int[] layerparams = {16, 16, OUTPUT_SIZE};
        MLP mlp = new MLP(28 * 28, layerparams);

        Value[] params = mlp.parameters(); // caching it
        saver.loadModel("MNIST_MODEL2", params);

        System.out.println(params.length);
        m = new double[params.length];
        v = new double[params.length];
        trainingLoop(mlp);

        saver.saveModel("MNIST_MODEL2", mlp.parameters());
        saver.saveAccuracy(acc);
    }

    public static void trainingLoop(MLP mlp) throws IOException {
        int howMany = (int) TOTAL_EPOCS / 10;

        int step = 0;
        int totalsteps = (int) 1e5;

        int counter = 0;
        int total_correct = 0;
        double loss;

        offsetData(howMany);
        do {
            mlp.zeroGrad();
            // get Images + Labels;
            Value[][] expectedLabels = getLabels(howMany);
            Value[][] images = getImages(howMany);

            // feed forward + softmax
            Value[][] preds = new Value[howMany][OUTPUT_SIZE];
            for (int i = 0; i < preds.length; i++) {
                preds[i] = softmax(mlp.call(images[i]));
            }

            // calc loss (L2 + Expected Loss)
            Value reg_loss = getMSEParams(mlp.parameters()).mult(new Value(ALPHA));
            Value data_loss = getCrossEntropyLoss(preds, expectedLabels);

            // backwards prop -> grads
            Value total_loss = data_loss.add(reg_loss);
            loss = total_loss.data;
            total_loss.backwards();

            // grad descent + adam https://arxiv.org/pdf/1412.6980
            double lrT = LEARNING_RATE * (1.0 - (double) step / totalsteps);
            ADAM(mlp, lrT, step);
            step++;
            // grade them + display
            for (int i = 0; i < images.length; i++) {
                counter++;
                if (getMaxValueIndex(preds[i]) == getMaxValueIndex(expectedLabels[i])) {
                    total_correct++;
                }
            }

            if (counter % TOTAL_EPOCS == 0) {
                displayValueImage(images[0]);
                System.out.println(Arrays.toString(expectedLabels[0]));
                System.out.println(Arrays.toString(preds[0]));
                System.out.println();
                printInfo(getAccuracy(total_correct, counter), data_loss, reg_loss, preds, expectedLabels);
                acc.add(getAccuracy(total_correct, counter));
                counter = 0;
                total_correct = 0;
            }
        } while (step < TOTAL_EPOCS * 10); // loss >= 1e-3
    }

    public static void offsetData(int howMany) throws IOException {
        for (int i = 0; i < (int) Math.random() * 1000; i++) {
            getLabels(howMany);
            getImages(howMany);
        }
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

    public static void printInfo(double accuracy, Value data_loss, Value reg_loss, Value[][] data, Value[][] expected) {
        for (int i = 0; i < data.length; i++) {
            System.out.printf("accuracy: %2.2f %% | loss: %.5f | reg_loss: %.5f | data: %s | data: %s | expected: %s\n", accuracy * 100, data_loss.data, reg_loss.data,
                    Arrays.toString(data[i]), getMaxValueIndex(data[i]), getMaxValueIndex(expected[i]));
        }
    }

    public static double getAccuracy(int correct, int total) {
        return (double) correct / total;
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

    public static void SGD(MLP mlp, double lrT) {
        Value[] params = mlp.parameters();

        for (int i = 0; i < params.length; i++) {
            params[i].data += lrT * -params[i].grad;
        }
    }

    public static void ADAM(MLP mlp, double lrT, int step) {
        double mCorrection = (1 - Math.pow(BETA_1, step + 1));
        double vCorrection = (1 - Math.pow(BETA_2, step + 1));
        Value[] params = mlp.parameters();

        for (int i = 0; i < params.length; i++) {
            Value param = params[i];
            m[i] = BETA_1 * m[i] + (1 - BETA_1) * param.grad;
            v[i] = BETA_2 * v[i] + (1 - BETA_2) * (param.grad * param.grad);

            double mHat = m[i] / mCorrection;
            double vHat = v[i] / vCorrection;

            param.data += -lrT * mHat / (Math.sqrt(vHat) + EPSILON);
        }
    }
}
