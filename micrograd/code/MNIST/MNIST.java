package micrograd.MNIST;
public class MNIST {

    // hyper parameters
    public static double LEARNING_RATE = 1e-1;
    public static double ALPHA = 1e-2;

    public static long TOTAL_EPOCS = (long) 1e4;

    public static void main(String[] args) {
        int[] layerparams = {16, 16, 3};
        MLP m = new MLP(784, layerparams);
    }

    public static void trainingLoop() {

    }

    public static void readData() {
        
    }
}
