
import java.util.Arrays;

public class ValueTesting {

    public static Value[] expected = {new Value(67), new Value(70)};

    public static void main(String[] args) {
        // ValueTesting();
        NeuronValueTesting();
        // gradientNumericalTest();

    }

    public static void NeuronValueTesting() {
        Value[][] inputs = {{new Value(5.1), new Value(1.1)},
        {new Value(.1), new Value(1.1)}};
        int[] layerparams = {16, 16, 3};
        MLP m = new MLP(inputs[0].length, layerparams);

        double lr = 0.5;
        double alpha = 1e-2;

        for (Value[] v : inputs) {
            System.out.println(Arrays.toString(m.call(v)));
        }

        Value[][] preds = new Value[inputs.length][inputs[0].length];
        do {
            m.zeroGrad();

            for (int i = 0; i < preds.length; i++) {
                preds[i] = m.call(inputs[i]);
            }

            // L2 norm for parameters
            Value reg_loss = getMeanSquaredParams(m.parameters()).mult(new Value(alpha));
            Value data_loss = getMSEloss(preds);

            Value total_loss = data_loss.add(reg_loss);
            total_loss.backwards();
            System.out.printf("\nloss: %.2f | reg_loss: %.2f | data: %s ", data_loss.data, reg_loss.data, Arrays.deepToString(preds));

            for (Value param : m.parameters()) {
                param.data += lr * -param.grad;
            }
        } while (getMSEloss(preds).data >=  1e-2);
        
        // System.out.println(Arrays.toString(m.parameters()));
        // System.out.println(Arrays.toString(m.call(inputs)));

    }

    public static Value getloss(Value[] outputs) {
        Value acc = new Value(0);

        for (int i = 0; i < outputs.length; i++) {
            // 1 - math.abs(outputs[i] - expected[i])/expected[i]
            Value calc = (((outputs[i].sub(expected[i])).abs()).div(expected[i]));
            acc = acc.add(calc);
        }
        return acc.div(new Value(outputs.length));
    }

    public static Value getMSEloss(Value[][] outputs) {
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

    public static Value getMeanSquaredParams(Value[] params) {
        Value output = new Value(0);
        for(Value p : params) {
            output = output.add(p.pow(2));
        }
        return output.div(new Value(params.length));
    }

    public static void ValueTesting() {
        Value a = new Value(-50);
        a.label = "a";
        Value b = a.RELU();
        b.label = "b";
        b.backwards();
        System.out.print(b);
    }

    public static void gradientNumericalTest() {
        double h = 0.001;

        Value a = new Value(1);
        a.label = "a";
        Value b = new Value(5);
        b.label = "b";
        Value c = a.add(b);
        c.label = "c"; // 6 
        Value d = new Value(2);
        d.label = "d";
        Value f1 = c.mult(d);
        f1.label = "f1"; // 6 + 2

        a = new Value(1);
        b = new Value(5);
        c = a.add(b); // 6
        d = new Value(2 + h);

        Value f2 = c.mult(d); // 6 + 2
        // f = C + D
        System.out.printf("Numerical Gradient: %.3f", (f2.data - f1.data) / h);
    }
}
