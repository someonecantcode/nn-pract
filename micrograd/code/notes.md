# How to read files

When reading ubyte files, we will read byte by byte, somtimes stitching them if needed.
We will use the `java.io.FileInputStream` and pass through our file path and use the methods.

Taken from https://docs.oracle.com/javase/8/docs/api/?java/io/FileInputStream.html:
| Methods     | Return | Description |
| ----------- | ----------- | --------- |
| read()      | int       |  	Reads a byte of data from this input stream |
| read(byte[] b) | int | Reads up to b.length bytes of data from this input stream into an array of bytes. |
| skip(long n)   |    long     | Skips over and discards n bytes of data from the input stream.|

## IDX 3


 Source - https://stackoverflow.com/a/54111674
 Posted by Charles Duffy, modified by community. See post 'Timeline' for change history Retrieved 2026-02-27, License - CC BY-SA 4.0

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images 


```java
public static final String IMAGE_PATH = "./data/t10k-images.idx3-ubyte";
this.fileReader = new FileInputStream(DataLoader.IMAGE_PATH);

```