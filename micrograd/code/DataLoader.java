
import java.io.FileInputStream;
import java.io.IOException;

public class DataLoader {

    public static final int ROWS = 28;
    public static final int COLUMNS = 28;
    public static final int SIZE = ROWS * COLUMNS;

    public static final String CHAR_DENSITY_GRADIENT = " .:-=+*#%@";

    public FileInputStream fileReader;
    public static final String IMAGE_PATH = "./data/t10k-images.idx3-ubyte";

    // taken from bhoener for testing purposes.
    public static void showImage(byte[] buffer) {
        // loop through rows and columns
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLUMNS; j++) {
                // print out the actual numerical value
                System.out.printf("%4d", buffer[i * COLUMNS + j] & 0xFF);
            }
            System.out.println();
        }
    }

    public DataLoader() throws IOException {
        this.fileReader = new FileInputStream(DataLoader.IMAGE_PATH);
    }

    public void readData() throws IOException {
        int totalstuff = this.fileReader.available();
        System.out.println(totalstuff);

        /**
         *
         * TRAINING SET IMAGE FILE (train-images-idx3-ubyte): [offset] [type]
         * [value] [description] 0000 32 bit integer 0x00000803(2051) magic
         * number 0004 32 bit integer 60000 number of images 0008 32 bit integer
         * 28 number of rows 0012 32 bit integer 28 number of columns
         *
         */
        for (int i = 0; i < 4; i++) { // first 16 bits header data (big endian)
            //int m = (this.fileReader.read());
            System.out.println((fileReader.read() << 24) | (fileReader.read() << 16) | (fileReader.read() << 8) | (fileReader.read()));
        }
        // this.fileReader.skip(784);
        // for(int i = 0; i < ROWS; i++){ // first image
        //     for(int j = 0; j < COLUMNS; j++) {
        //         int m = (this.fileReader.read());
        //         System.out.printf("%3s ", m);
        //     }
        //     System.out.println();
        // }

    }

    public void closefileReader() throws IOException {
        this.fileReader.close();
    }
}


/*
create data structure with O(1) min, max peek.

inorder traversal of a binary search tree returns a sorted array of numbers.
breadth first search is inefficient because you do not know the children usually and it takes time to calculate.
 */
