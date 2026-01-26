import java.util.ArrayList;

class Node {
    String data; 
    ArrayList<Node> children;
    public Node(String data) {
        this.data = data;
        this.children = new ArrayList<>();
    }

    public Node addChildrenNode(String data) {
        Node next = new Node(data);
        children.add(next);
        return next;
    }

    public void addChildrenNode(Node n) {
        children.add(n);
    }

    @Override
    public String toString() {
        return String.format("%s", this.data);
    }
}

class Graph {
    Node[] nodes;

    public Graph(){

    }
}

public class DFS {

    public static void main(String[] args) {
        testing();
    }

    public static void testing() {
        Node A = new Node("A");
        Node B = A.addChildrenNode("B");
        Node C = A.addChildrenNode("C");

        B.addChildrenNode(C);

        System.out.println(C);
        System.out.println(C.children);

        algorithm(A);
    }

    public static Node[] algorithm(Node input){
        return new Node[] {};
    }
}