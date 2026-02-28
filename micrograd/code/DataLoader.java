
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

public class DataLoader {

    private static final int IMAGE_SIZE = 784;
    private static final String CHAR_DENSITY_GRADIENT = "`.:-=+*#%@";

    private final FileInputStream LabelReader; //idx1 = 2049
    private final FileInputStream ImageReader; //idx3 = 2051

    private int[] LabelHeader = new int[2];
    private int[] ImageHeader = new int[4];

    public DataLoader(String LABEL_PATH, String IMAGE_PATH) throws IOException {
        this.LabelReader = new FileInputStream(LABEL_PATH);
        this.ImageReader = new FileInputStream(IMAGE_PATH);

        this.getHeader(this.LabelReader, this.LabelHeader);
        this.getHeader(this.ImageReader, this.ImageHeader);
    }

    public void readDataTEST() throws IOException {
        // int totalstuff = this.ImageReader.available();
        // System.out.println(totalstuff);
        System.out.println(Arrays.toString(LabelHeader));
        System.out.println(Arrays.toString(ImageHeader));

        for (int n = 0; n < 3; n++) {
            displayImage(readImage());
            System.out.println(readLabel());
        }

    }

    public int readLabel() throws IOException {
        return (this.LabelReader.read());
    }

    public int[] readImage() throws IOException {
        // [magicnumber, #images, row, col]
        final int ROW_INDEX = 2;
        final int COL_INDEX = 3;

        int[] imageOutput = new int[ImageHeader[ROW_INDEX] * ImageHeader[COL_INDEX]];
        for (int i = 0; i < imageOutput.length; i++) {
            imageOutput[i] = this.ImageReader.read();
        }

        return imageOutput;
    }

    public void getHeader(FileInputStream fileReader, int[] header) throws IOException {
        for (int i = 0; i < header.length; i++) { // to fill the array
            header[i] = (fileReader.read() << 24) | (fileReader.read() << 16) | (fileReader.read() << 8) | (fileReader.read());
        }
    }

    public void closefileReader() throws IOException {
        this.ImageReader.close();
        this.LabelReader.close();
    }

    private void displayImage(int[] imageOutput) {
        final int ROW_INDEX = 2;
        final int COL_INDEX = 3;

        assert imageOutput.length == (ImageHeader[COL_INDEX] * ImageHeader[ROW_INDEX]) : "Image length not 784";

        for (int i = 0; i < ImageHeader[COL_INDEX]; i++) { // first image
            for (int j = 0; j < ImageHeader[ROW_INDEX]; j++) {
                int m = imageOutput[i*ImageHeader[ROW_INDEX]+j];

                char brightness = CHAR_DENSITY_GRADIENT.charAt((m * (CHAR_DENSITY_GRADIENT.length() - 1)) / (255));
                System.out.printf("%3c", brightness);
            }
            System.out.println();
        }
    }
}


/*
create data structure with O(1) min, max peek.

inorder traversal of a binary search tree returns a sorted array of numbers.
breadth first search is inefficient because you do not know the children usually and it takes time to calculate.
 */
