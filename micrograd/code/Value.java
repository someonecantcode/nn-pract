
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

// refer to https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
public class Value {

    double data;
    double grad;
    ArrayList<Value> prev_children;
    String label = "defaultlabel";

    private Runnable _backward = () -> {
    };

    public Value(double data, ArrayList<Value> prev_children) {
        this.data = data;
        this.prev_children = prev_children;
        this.grad = 0;
    }

    public Value(double data) {
        this(data, new ArrayList<Value>());
    }

    private ArrayList<Value> inputsToChildren(Value b) {
        return new ArrayList<Value>(Arrays.asList(this, b));
    }

    public Value add(Value b) {
        Value out = new Value(this.data + b.data, inputsToChildren(b));
        /*
        C = A + B
        del C = 1
        del A

        f(x) = a(x) + b(x)
        del f/del a = del f/del f * del a/del b

         */
        out._backward = () -> {
            this.grad += 1 * out.grad;
            b.grad += 1 * out.grad;
        };

        return out;
    }

    public Value mult(Value b) {
        Value out = new Value(this.data * b.data, inputsToChildren(b));
        out._backward = () -> {
            this.grad += b.data * out.grad;
            b.grad += this.data * out.grad;
        };
        return out;
    }

    public Value neg(Value b) {
        b.data = -1 * b.data;
        return b;
    }

    public Value sub(Value b) { // a - b
        return this.add(neg(b));
    }

    public Value pow(Value b) {
        Value out = new Value(Math.pow(this.data, b.data));
        out._backward = () -> {
            this.grad += (b.data * Math.pow(this.data, b.data - 1)) * out.grad;
            b.grad += (this.data * Math.log(b.data) * out.data) * out.grad;
        };
        return out;
    }

    // public Value div(Value b) {
    //     return this.mult(b.pow(neg(new Value(1))));
    // }
    public void backwards() {
        ArrayList<Value> topo = new ArrayList<>(this.prev_children.size());
        HashSet<Value> visited = new HashSet<>();

        DFS(this, topo, visited);
        this.grad = 1;

        Collections.reverse(topo);
        for (Value v : topo) {
            v._backward.run();
        }
        System.out.print("Backward Propogation done. OUTPUT: ");
        System.out.println(topo);

    }

    private static void DFS(Value v, ArrayList<Value> topo, HashSet<Value> visisted) {
        if (!visisted.contains(v)) {
            visisted.add(v);
            for (Value child : v.prev_children) {
                DFS(child, topo, visisted);
            }

            topo.add(v);
        }

    }

    @Override
    public String toString() {
        String output = String.format("Value %s: %.1f Gradient: %.1f", this.label, this.data, this.grad);
        return output;
    }

}
