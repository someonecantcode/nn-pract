

public class ValueTesting {

    public static void main(String[] args) {
        // myTesting();
        // ValueTesting();
        NeuronValueTesting();
        // gradientNumericalTest();

    }

    public static void myTesting() {
       
    }

    public static void NeuronValueTesting() {
        double[] inputs = {1, 2};
        Neuron n = new Neuron(inputs.length, true);

        double lr = 0.001;
        while(n.call(inputs).data <= 1000) {
            n.zeroGrad();
    
            
            Value call =  n.call(inputs);
            call.backwards();
            System.out.println(call);
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
