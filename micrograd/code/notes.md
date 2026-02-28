# How to read IDX-UBYTE files

When reading ubyte files, we will read byte by byte, somtimes stitching them if needed.
We will use the `java.io.FileInputStream` and pass through our file path and use the methods.

Taken from https://docs.oracle.com/javase/8/docs/api/?java/io/FileInputStream.html:
| Methods     | Return | Description |
| ----------- | ----------- | --------- |
| read()      | int       |  	Reads a byte of data from this input stream |
| read(byte[] b) | int | Reads up to b.length bytes of data from this input stream into an array of bytes. |
| skip(long n)   |    long     | Skips over and discards n bytes of data from the input stream.|
| available()   | int | Returns an estimate of the number of remaining bytes that can be read (or skipped over) from this input stream without blocking by the next invocation of a method for this input stream. |

```java
FileInputStream reader = new FileInputStream(String FilePath);
reader.read();
```

## IDX 1 (Labels)

Header

```
[magicIdentifier, number of images, row, column]

magicIdentifer = 2049
number of labels = 10000
```

```java
public static final String LABEL_PATH = "./data/t10k-images.idx3-ubyte";
FileInputStream LabelReader = new FileInputStream(LABEL_PATH);

// header
private int[] LabelHeader = new int[2];
for (int i = 0; i < ImageHeader.length; i++) { // to fill the array
    header[i] = (fileReader.read() << 24) | (fileReader.read() << 16) | (fileReader.read() << 8) | (fileReader.read());
}
```

## IDX 3 (Images)

 Source - https://stackoverflow.com/a/54111674
 Posted by Charles Duffy, modified by community. See post 'Timeline' for change history Retrieved 2026-02-27, License - CC BY-SA 4.0

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images 

Header
```
[magicIdentifier, number of images, row, column]

magicIdentifer = 2051
number of images = 10000
row = 28
col = 29
```

```java
public static final String IMAGE_PATH = "./data/t10k-images.idx3-ubyte";
FileInputStream ImageReader = new FileInputStream(IMAGE_PATH);

// header
private int[] ImageHeader = new int[4];
for (int i = 0; i < ImageHeader.length; i++) { // to fill the array
    header[i] = (fileReader.read() << 24) | (fileReader.read() << 16) | (fileReader.read() << 8) | (fileReader.read());
}

public void readImage() throws IOException {
    // [magicnumber, #images, row, col]
    final int ROW_INDEX = 2;
    final int COL_INDEX = 3;

    System.out.println(Arrays.toString(ImageHeader));

    for (int i = 0; i < ImageHeader[ROW_INDEX]; i++) { // first image
        for (int j = 0; j < ImageHeader[COL_INDEX]; j++) {
            int m = (this.ImageReader.read());
            char brightness = CHAR_DENSITY_GRADIENT.charAt((m * (CHAR_DENSITY_GRADIENT.length() - 1)) / (255));
            System.out.printf("%3c", brightness);
        }
        System.out.println();
    }
}
```


