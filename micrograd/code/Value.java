import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

public class Value {
    double data;
    double grad = 0;
    ArrayList<Value> prev_children;

    public Value(double data, ArrayList<Value> prev_children) {
        this.data = data;
        this.prev_children = prev_children;
    }

    public Value(double data) {
        this(data, new ArrayList<Value>());
    }

    public Value add(Value b){
        
        // void backwards() {
        //     this.grad += b.data;
        //     b.grad += this.data;
        // }

        return new Value(this.data + b.data);
    }

    public Value neg(Value a){
        return new Value(-1 * a.data);
    }

    public Value sub(Value b){ // a - b
        return this.add(neg(b));
    }

    public Value mult(Value b){
        return new Value(this.data * b.data);
    }

    public Value pow(Value b){
        return new Value(Math.pow(this.data, b.data));
    }

    public Value div(Value b){
        return this.mult(b.pow(neg(new Value(1))));
    }


    public void backwards(){
        ArrayList<Value> topo = new ArrayList<>(this.prev_children.size());
        HashSet<Value> visited = new HashSet<>();

        DFS(this, topo, visited);
        this.grad = 1;

        Collections.reverse(topo);
        for(Value v : topo){
            v.backwards();
        }
    
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
    public String toString(){
        String output = String.format("Value:    %f \nGradient: %f", this.data, this.grad);
        return output;
    }

}