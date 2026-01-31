import java.util.Arrays;

public class ValueTesting {

    public static void main(String[] args) {
        // ValueTesting();
        NeuronValueTesting();
        // gradientNumericalTest();

    }

    public static void NeuronValueTesting() {
        double[] inputs = {5, 15, 20};
        LayerValue n = new LayerValue(inputs.length, 2, false);

        double lr = 0.1;
        while(n.call(inputs)[0].data <= 1000) {
            n.zeroGrad();

            Value[] outputs = n.call(inputs);
            for(Value out : outputs){
                out.backwards();
            }
            System.out.println(Arrays.toString(outputs));
            //System.out.println(Arrays.toString(n.parameters()));
            //System.out.println();
            for (Value param : n.parameters()) {
                param.data += lr * param.grad;
            }
            
        }
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
