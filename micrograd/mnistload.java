import java.io.File;
import java.io.IOException;

public class DataLoader {
    public static final int ROWS = 28;
    public static final int COLUMNS = 28;
    public static final int SIZE = ROWS * COLUMNS;

    public static final String CHAR_DENSITY_GRADIENT = " .:-=+*#%@";
    public static final String IMAGE_PATH = "../data/"


    public DataLoader() throws IOException {

    }

    // taken from bhoener for testing purposes.
    public static void showImage(byte[] buffer) {
        // loop through rows and columns
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // print out the actual numerical value
                System.out.printf("%4d", buffer[i * cols + j] & 0xFF);
            }
            System.out.println();
        }
    }

    public static void readData() {
        
    }


    public static void main(String[] args) {

    }
}
