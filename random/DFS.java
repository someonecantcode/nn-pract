
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

class Node {

    String data;
    ArrayList<Node> prev_children;

    public Node(String data) {
        this.data = data;
        this.prev_children = new ArrayList<>();
    }

    public Node addChildrenNode(String data) {
        Node next = new Node(data);
        prev_children.add(next);
        return next;
    }

    public void addChildrenNode(Node n) {
        prev_children.add(n);
    }

    @Override
    public String toString() {
        return String.format("%s", this.data);
    }
}

public class DFS {

    public static void main(String[] args) {
        testing();
    }

    public static void testing() {
        Node A = new Node("A");
        Node C = new Node("C");

        Node B = new Node("B");
        B.addChildrenNode(A);
        B.addChildrenNode(C);

        Node E = new Node("E");
        Node D = new Node("D");
        D.addChildrenNode(B);
        D.addChildrenNode(E);

        System.out.println(D);
        System.out.println(D.prev_children);

        System.out.println(algorithm(D));
    }

    private static ArrayList<Node> algorithm(Node input) {
        // dfs works by going deep until there are no more children 
        // backtracking and repeating. we implement it recursively with the following

        ArrayList<Node> topo = new ArrayList<>(input.prev_children.size());
        HashSet<String> visited = new HashSet<>();

        DFS(input, topo, visited);

        Collections.reverse(topo); // root -> children for back prop
        return topo;
    }

    private static void DFS(Node n, ArrayList<Node> topo, HashSet<String> visisted) {
        if (!visisted.contains(n.data)) {
            visisted.add(n.data);
            for (Node child : n.prev_children) {
                DFS(child, topo, visisted);
            }

            topo.add(n);
        }

    }
}
