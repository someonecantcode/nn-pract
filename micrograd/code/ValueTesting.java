
import java.util.Arrays;

public class ValueTesting {

    public static void main(String[] args) {
        // ValueTesting();
        NeuronValueTesting();
        // gradientNumericalTest();

    }

    public static void NeuronValueTesting() {
        double[] inputs = {0.4, 0.1};
        LayerValue n = new LayerValue(inputs.length, 2, false);

        double lr = 0.5;
        System.out.println(Arrays.toString(n.call(inputs)));

        while (getloss(n.call(inputs)).data >= .01) {
            n.zeroGrad();

            Value[] outputs = n.call(inputs);
            Value loss = getloss(outputs);

            loss.backwards();
            // System.out.println(Arrays.toString(outputs));
            System.out.println(1 - loss.data);

            for (Value param : n.parameters()) {
                param.data += lr * -param.grad;
            }
        }
        System.out.println(Arrays.toString(n.call(inputs)));
        System.out.println(Arrays.toString(n.parameters()));
    }

    public static Value getloss(Value[] outputs) {
        Value acc = new Value(0);
        double[] expected = {50, 70};
        for (int i = 0; i < outputs.length; i++) {
            // 1 - math.abs(outputs[i] - expected[i])/expected[i]
            Value calc = (((outputs[i].sub(new Value(expected[i]))).abs()).div(new Value(expected[i])));
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
