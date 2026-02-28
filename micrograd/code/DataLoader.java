
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.classfile.Label;
import java.util.Arrays;

public class DataLoader {

    private static final int IMAGE_SIZE = 784;
    private static final String CHAR_DENSITY_GRADIENT = " .:-=+*#%@";

    private final FileInputStream LabelReader; //idx1
    private final FileInputStream ImageReader; //idx3
    

    public DataLoader(String IMAGE_PATH, String LABEL_PATH) throws IOException {
        this.LabelReader = new FileInputStream(LABEL_PATH);
        this.ImageReader = new FileInputStream(IMAGE_PATH);
    }

    public void readDataTEST() throws IOException {
        int totalstuff = this.ImageReader.available();
        System.out.println(totalstuff);

        int[] headerValues = getHeader(this.ImageReader);
        System.out.println(Arrays.toString(headerValues));

        for (int n = 0; n < 10; n++) {
            readLabel(headerValues);
            //   readImage(headerValues);
        }

    }

    public void readLabel(int[] headerValues) throws IOException {
        int m = (this.LabelReader.read());
        System.out.println(m);
    }
    
    public void readImage() throws IOException {
        // [magicnumber, #images, row, col]
        final int ROW_INDEX = 2;
        final int COL_INDEX = 3;

        int[] headerValues = getHeader(ImageReader);

        for (int i = 0; i < headerValues[ROW_INDEX]; i++) { // first image
            for (int j = 0; j < headerValues[COL_INDEX]; j++) {
                int m = (this.ImageReader.read());
                char brightness = CHAR_DENSITY_GRADIENT.charAt((m * (CHAR_DENSITY_GRADIENT.length() - 1)) / (255));
                System.out.printf("%3c", brightness);
            }
            System.out.println();
        }
    }


    public int[] getHeader(FileInputStream fileReader) throws IOException {
        int magicIdentifer = (fileReader.read() << 24) | (fileReader.read() << 16) | (fileReader.read() << 8) | (fileReader.read()); //magic identifier idx1= 2049, idx=2051
        int headerSize = getHeaderSizeIDX(magicIdentifer);

        int[] headerValues = new int[headerSize];
        headerValues[0] = magicIdentifer;
        for (int i = 1; i < headerSize; i++) { // to fill the array
            headerValues[i] = (fileReader.read() << 24) | (fileReader.read() << 16) | (fileReader.read() << 8) | (fileReader.read());
        }
        return headerValues;
    }

    public int getHeaderSizeIDX(int magicIdentifer) {
        // if (magicIdentifer == 2049) {
        //     return 2;
        // }

        // if (magicIdentifer == 2051) {
        //     return 4;
        // }
        return magicIdentifer - 2047;
    }

    public void closefileReader() throws IOException {
        this.ImageReader.close();
        this.LabelReader.close();
    }
}


/*
create data structure with O(1) min, max peek.

inorder traversal of a binary search tree returns a sorted array of numbers.
breadth first search is inefficient because you do not know the children usually and it takes time to calculate.
 */
