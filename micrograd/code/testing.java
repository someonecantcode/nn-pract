
import java.util.Arrays;

public class testing {

    public static void main(String[] args) {
        //myTesting();


    }

    public static void myTesting() {
        double[] testarray;
        double[] a1 = {1, 2, 3, 4, 5};

        testarray = Arrays.copyOf(a1, a1.length); // idc thats it referenced
        System.out.println(Arrays.toString(testarray));
        double[] a2 = {1, 2, 3};

        testarray = Arrays.copyOf(a2, a2.length); // idc thats it referenced
        System.out.println(Arrays.toString(testarray));


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
        Value a = new Value(1);
        Value b = new Value(5);

        Value c = a.add(b);
        System.out.println(c.grad);
    }
}
