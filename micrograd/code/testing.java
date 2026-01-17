import java.util.Arrays;

public class testing {

    public static void main(String[] args) {
        myTesting();
        // double[] inputs = {1, 2, 3, 4, 5};


        // Layer l = new Layer(5, 2, true); // 5 inputs 2 neurons

        // System.out.println(Arrays.toString(l.call(inputs)));
        // System.out.println(Arrays.deepToString(l.parameters()));
        // System.out.println(l);
    }

    public static void myTesting(){
        double[] testarray;
        double[] a1 = {1, 2, 3, 4, 5};

        testarray = Arrays.copyOf(a1, a1.length); // idc thats it referenced
        System.out.println(Arrays.toString(testarray));
        double[] a2 = {1, 2, 3};

        testarray = Arrays.copyOf(a2, a2.length); // idc thats it referenced
        System.out.println(Arrays.toString(testarray));
    }

    public static void ValueTesting() {
        Value a = new Value(1);
        Value b = new Value(5);

        Value c = a.add(b);
        System.out.println(c.grad);
    }
}
