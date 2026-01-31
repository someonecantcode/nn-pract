
import java.util.Arrays;

public class ValueTesting {

    public static void main(String[] args) {
        // myTesting();
       // ValueTesting();
        // gradientNumericalTest();
        Value a = new Value(-50); a.label = "a";
        Value b = a.RELU(); b.label = "b";
        b.backwards();
        System.out.print(b);
    }

    public static void myTesting() {
        double[] inputs = {1, 2, 3, 4, 5};

        int[] layerparams = {16, 16, 1};
        for (int i = 0; i < 1; i++) {
            MLP m = new MLP(inputs.length, layerparams); // 5 inputs 2 neurons

            // System.out.println(m);
            System.out.printf("params: %s", Arrays.deepToString(m.parameters()));
            // System.out.println(Arrays.toString(m.call(inputs)));

            System.out.println();
        }
    }

    public static void ValueTesting() {
        double[] inputs = {1};
        NeuronValue n = new NeuronValue(inputs.length, true);

        // System.out.println(Arrays.toString(n.parameters()));
       System.out.println(n.call(inputs));
       n.call(inputs).backwards();
    }

    public static void gradientNumericalTest() {
        double h = 0.001;

        Value a = new Value(1); a.label = "a";
        Value b = new Value(5); b.label = "b";
        Value c = a.add(b); c.label = "c"; // 6 
        Value d = new Value(2); d.label = "d";
        Value f1 = c.mult(d); f1.label = "f1"; // 6 + 2

        a = new Value(1);
        b = new Value(5);
        c = a.add(b); // 6
        d = new Value(2 + h);

        Value f2 = c.mult(d); // 6 + 2
        // f = C + D
        System.out.printf("Numerical Gradient: %.3f",(f2.data - f1.data) / h);
    }
}
