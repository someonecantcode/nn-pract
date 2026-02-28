package nnlib;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

// refer to https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
public class Value {

    public double data;
    public double grad;
    public String label = "dfl";
    ArrayList<Value> prev_children;

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

    public Value twovarpow(Value b) { // a^b. if a is negative, then the gradients might break
        Value out = new Value(Math.pow(this.data, b.data), inputsToChildren(b));
        out._backward = () -> {
            this.grad += (b.data * Math.pow(this.data, b.data - 1)) * out.grad;
            b.grad += (Math.log(this.data) * out.data) * out.grad;
        };
        return out;
    }

    public Value pow(double b) {
        Value out = new Value(Math.pow(this.data, b));

        out.prev_children.add(this);
        out._backward = () -> {
            this.grad += (b * Math.pow(this.data, b - 1)) * out.grad;
        };
        return out;
    }

    public Value RELU() {
        Value out = new Value(Math.max(0, this.data));

        out.label = "RELU";
        out.prev_children.add(this);
        out._backward = () -> {
            if (out.data > 0) {
                this.grad += 1 * out.grad;
            }
            // if (out.data < 0) this.grad += 0 * out.grad; 
            // everything else is all grad = 0;
        };
        return out;
    }

    public Value tanh() {
        Value out = new Value(Math.tanh(this.data));

        out.label = "tanh";
        out.prev_children.add(this);
        out._backward = () -> {
            this.grad += (1 - Math.pow(out.data, 2))* out.grad;
        };
        return out;
    }

    public Value abs() {
        Value out = new Value(Math.abs(this.data));

        out.label = "abs";
        out.prev_children.add(this);
        out._backward = () -> {
            if (this.data > 0 ){
                 this.grad += 1 * out.grad;
            }

             if (this.data < 0 ){
                 this.grad += -1 * out.grad;
            }
           
        };
        return out;
    }

    public Value div(Value b) {
        return this.mult(b.pow(-1));
    }

    public Value neg() {
        return this.mult(new Value(-1));
    }

    public Value sub(Value b) { // a - b
        return this.add(b.neg());
    }

    public void backwards() {
        ArrayList<Value> topo = new ArrayList<>(this.prev_children.size());
        HashSet<Value> visited = new HashSet<>();

        DFS(this, topo, visited);
        this.grad = 1;

        Collections.reverse(topo);
        for (Value v : topo) {
            v._backward.run();
        }
        // System.out.print("Backward Propogation done. OUTPUT: ");
        // System.out.println(topo);
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
        // if (this.grad == 0) {
        //     return "";
        // }

        if (this.data == -1) {
            return "negative node";
        }
        String output = String.format("Value %s: %5.1f Gradient: %5.4f", this.label, this.data, this.grad);
        return output;
    }

}
