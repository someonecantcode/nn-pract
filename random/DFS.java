import java.util.ArrayList;

class Node {
    String data; 
    ArrayList<Node> children;
    public Node(String data) {
        this.data = data;
        this.children = new ArrayList<>();
    }

    public Node nextNode(String data) {
        Node next = new Node(data);
        children.add(next);
        return next;
    }
}

public class DFS {

    public static void main(String[] args) {
        
    }

    public Node[] algorithm(Node[] input){
        return new Node[] {};
    }
}