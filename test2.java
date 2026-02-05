class X {
    int i = 1;
    X() {
        System.out.print("X");
    }
}

class Y extends X {
    int i = 3;
    Y() {
        super.i++;
        this.i += super.i;
        System.out.print("Y");
    }
}


public class test2 {
    public static void main(String[] args) {
        X xy = new Y();
        System.out.println(xy.i);
    }
}