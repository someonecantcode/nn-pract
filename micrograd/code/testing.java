import java.util.Arrays;

public class testing {

    public static void main(String[] args) {
        double[] inputs = {1, 2, 3, 4, 5};
        Layer l = new Layer(5, 2, true); // 5 inputs 2 neurons

        System.out.println(Arrays.toString(l.call(inputs)));
        // System.out.println(Arrays.deepToString(l.parameters()));
        // System.out.println(l);
    }

    public static void ValueTesting() {
        Value a = new Value(1);
        Value b = new Value(5);

        Value c = a.add(b);
        System.out.println(c.grad);
    }
}
