import java.io.IOException;
import nnlib.MLP;

public class MNIST {

    // hyper parameters
    public static double LEARNING_RATE = 1e-1;
    public static double ALPHA = 1e-2;

    public static long TOTAL_EPOCS = (long) 1e4;

    public static void main(String[] args) throws IOException {
        testing();
        
        int[] layerparams = {16, 16, 3};
        MLP m = new MLP(28*28, layerparams);

        //  System.out.println(m.parameters().length);
    }

    public static void trainingLoop() {

    }

    public static void testing() throws IOException {
        DataLoader d = new DataLoader();
        d.readData();
    }

}
