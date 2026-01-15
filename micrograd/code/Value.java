

public class Value {
    double data;
    double grad = 0;

    public Value(double data) {
        this.data = data;
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


    // public void backwards(){
    //     ArrayList<Integer> topo = new ArrayList<>();
    //     HashSet<Integer> visited = new HashSet<>();
    //     void build_topo(Value a){
    //         if v not in visited{
    //             visited.add(v)
    //             for child in v._prev:
    //                 build_topo(child)
    //             topo.append(v)
    //         }    
    //     }
    //     build_topo(self)

    //     this.grad = 1;
    //     for v in reversed(topo):
    //         v._backward()
    // }

    @Override
    public String toString(){
        String output = String.format("Value:    %f \nGradient: %f", this.data, this.grad);
        return output;
    }

}