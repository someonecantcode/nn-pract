
import java.util.Arrays;

public class testing {

    public static void main(String[] args) {
        //myTesting();
        // ValueTesting();
        gradientNumericalTest();
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

        // f.backwards();
        // System.out.println(f);
        // System.out.println(f.prev_children);
    }

    public static void gradientNumericalTest() {
        double h = 0.001;

        Value a = new Value(1);
        Value b = new Value(5);
        Value c = a.add(b); // 6
        Value d = new Value(2);
        Value f1 = c.mult(d); // 6 + 2

        a = new Value(1);
        b = new Value(5);
        c = a.add(b); // 6
        d = new Value(2 + h);

        Value f2 = c.mult(d); // 6 + 2
        // f = C + D
        System.out.println((f2.data - f1.data) / h);
    }
}
