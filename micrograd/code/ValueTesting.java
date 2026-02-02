
import java.util.Arrays;

public class ValueTesting {
    public static Value[] expected = {new Value(500), new Value(70)};
    public static void main(String[] args) {
        // ValueTesting();
        NeuronValueTesting();
        // gradientNumericalTest();

    }

    public static void NeuronValueTesting() {
        Value[] inputs = {new Value(0.4), new Value(0.1)};
        int[] layerparams = {16, 16, 2};
        MLP m = new MLP(inputs.length, layerparams);

        double lr = 0.1;
        System.out.println(Arrays.toString(m.call(inputs)));

        while (getMSEloss(m.call(inputs)).data >= .01) {
            m.zeroGrad();

            Value[] outputs = m.call(inputs);
            Value loss = getMSEloss(outputs);

            loss.backwards();
            // System.out.println(Arrays.toString(outputs));
            System.out.printf("\nloss: %.6f | acc: %.6f | data: %s ", loss.data, 1 - getloss(outputs).data, Arrays.toString(outputs));

            for (Value param : m.parameters()) {
                param.data += lr * -param.grad;
            }
        }
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

    public static Value getMSEloss(Value[] outputs) {
        Value acc = new Value(0);

        for (int i = 0; i < outputs.length; i++) {
            // (expected[i] - output[i])^2
            Value calc = (expected[i].sub(outputs[i])).pow(2);
            acc = acc.add(calc);
        }
        return acc.div(new Value(outputs.length));
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
