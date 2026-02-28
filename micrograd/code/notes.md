# How to read files

 Source - https://stackoverflow.com/a/54111674
 Posted by Charles Duffy, modified by community. See post 'Timeline' for change history Retrieved 2026-02-27, License - CC BY-SA 4.0

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images 


```java
this.fileReader = new FileInputStream(DataLoader.IMAGE_PATH);

```