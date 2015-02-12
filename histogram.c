// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
#define LINEAR_WIDTH 256
#define BLOCK_WIDTH 16


__global__ void ucharImage_kernel(unsigned char *out, float *in, int imageHeight, int imageWidth, int imageChannels){
  int tx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tx;

  if (idx < imageHeight*imageWidth*imageChannels){
    out[idx] = (unsigned char) (255 * in[idx]);
  }
  //printf("in[idx]: %f; out[idx]: %d\n", in[idx], out[idx]);
}

__global__ void grayscaleImage_kernel( unsigned char *out, unsigned char *in, int imageHeight, int imageWidth, int imageChannels){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + tx;
  int row = blockIdx.y * blockDim.y + ty;

  int idx = row * imageWidth + col;
  if (idx < imageHeight*imageWidth*imageChannels){
    out[idx] = (unsigned char) (0.21*in[3*idx] + 0.71*in[3*idx + 1] + 0.07*in[3*idx + 2]);
  }
  //printf("out[idx]: %d\n", out[idx]);

}

int main(int argc, char ** argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float * deviceInputImageData;
  float * deviceOutputImageData;
  unsigned char * deviceUcharImageData;
  unsigned char * deviceGrayImageData;

  float * hostInputImageData;
  float * hostOutputImageData;

  const char * inputImageFile;




  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceUcharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGrayImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  //TODO all the temporary variables
  wbTime_stop(GPU, "Doing GPU memory allocation");

  //copying memory
  wbTime_start(Copy, "Copying data to GPU");
  cudaMemcpy(deviceInputImageData,
      hostInputImageData,
      imageWidth * imageHeight * imageChannels * sizeof(float),
      cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to GPU");

  int imageWidthGrid = (imageWidth-1)/BLOCK_WIDTH + 1;
  int imageHeightGrid = (imageHeight-1)/BLOCK_WIDTH + 1;
  //cast to unsigned char (image sized)
  dim3 castGrid(  (imageWidth * imageHeight * imageChannels - 1)/LINEAR_WIDTH + 1, 1, 1);
  dim3 castBlock( LINEAR_WIDTH, 1, 1);
  ucharImage_kernel<<<castGrid,castBlock>>>( deviceUcharImageData, deviceInputImageData,
      imageHeight, imageWidth, imageChannels
      );

  //create grayscale image (image sized)
  dim3 grayGrid( imageWidthGrid, imageHeightGrid, 1);
  dim3 grayBlock( BLOCK_WIDTH, BLOCK_WIDTH, 1);
  grayscaleImage_kernel<<<grayGrid, grayBlock>>> ( deviceGrayImageData, deviceUcharImageData,
      imageHeight, imageWidth, imageChannels
      );

  //Compute histogram (image sized input, 256 wide output)
  dim3 histGrid( imageWidthGrid, imageHeightGrid, 1);
  dim3 histBlock ( BLOCK_WIDTH, BLOCK_WIDTH, 1);

  //compute cdf (256 wide -- prefix sum)
  dim3 cdfGrid( HISTOGRAM_LENGTH, 1, 1);
  dim3 cdfBlock( 1,1,1 ); //it is so small can just do 1

  //reduce cdf (find min, 256 wide)
  dim3 minGrid( HISTOGRAM_LENGTH, 1, 1);
  dim3 minBlock( 1,1,1 );

  //apply histogram equalization on ucharimage (image sized)
  dim3 correctGrid( imageWidthGrid, imageHeightGrid, 1);
  dim3 correctBlock ( BLOCK_WIDTH, BLOCK_WIDTH, 1);

  //cast back to floating point image (image sized)
  dim3 floatGrid( imageWidthGrid, imageHeightGrid, 1);
  dim3 floatBlock ( BLOCK_WIDTH, BLOCK_WIDTH, 1);


  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData,
      deviceOutputImageData,
      imageWidth * imageHeight * imageChannels * sizeof(float),
      cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");


  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}


