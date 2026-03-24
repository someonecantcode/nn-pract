
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Scanner;
import nnlib.*;

public class ModelSaver {

    private final String MODEL_DIRECTORY = "./model/";

    public void loadModel(String modelName, Value[] parameters) throws FileNotFoundException {
        File model = new File(MODEL_DIRECTORY + modelName);
        if (!model.exists()) {
            System.out.println("NO MODEL FOUND. USING DEFAULT RANDOMIZED VALUES. EXITING");
            return;
        }

        Scanner scan = new Scanner(model);

        for (Value v : parameters) {
            v.data = scan.nextDouble();
        }
        scan.close();
    }

    public void saveModel(String modelName, Value[] parameters) throws FileNotFoundException {
        File fout = new File(MODEL_DIRECTORY + modelName);
        PrintStream writer = new PrintStream(fout);

        for (Value v : parameters) {
            writer.print(v.data + " ");
        }
        System.out.println("MODEL SAVED");
    }

    public void saveAccuracy(ArrayList<Double> accuracy) throws FileNotFoundException {
        File fout = new File(MODEL_DIRECTORY + "graph");
        PrintStream writer = new PrintStream(fout);

        for (double data : accuracy) {
            writer.print(data + " ");
        }
        System.out.println("accuracy saved");
    }
}
