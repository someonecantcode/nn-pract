

public class testing {

    public static void main(String[] args) {
        Value a = new Value(1);
        Value b = new Value(5);

        Value c = a.add(b);

        System.out.println(c.grad);

        double[] inputs = {1, 2, 3, 4, 5};
        Neuron n = new Neuron(inputs.length);
        System.out.println(n.call(inputs));
        System.out.println(n);

        System.out.println("\n Parameters:");
        for(double params : n.parameters()){
            System.out.println(params);
        }
    }
}
