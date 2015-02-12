// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
#define LINEAR_WIDTH 256
#define BLOCK_WIDTH 16

//TODO add stride = blockDim.x * gridDim.x if we exceed gridsize on super large pictures.
__global__ void ucharImage_kernel(unsigned char *out, float *in, int imageHeight, int imageWidth, int imageChannels){
  int tx = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tx;

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

  unsigned int idx = row * imageWidth + col;
  if (idx < imageHeight*imageWidth*imageChannels){
    out[idx] = (unsigned char) (0.21*in[3*idx] + 0.71*in[3*idx + 1] + 0.07*in[3*idx + 2]);
  }
  //printf("out[idx]: %d\n", out[idx]);

}

__global__ void histogram_kernel ( unsigned int *histogram, unsigned char *grayimg, unsigned int size ) {

  __shared__ unsigned int histogram_priv[256];

  int tx = threadIdx.x;
  unsigned int idx = tx + blockIdx.x * blockDim.x;

  if (tx < HISTOGRAM_LENGTH) {
    histogram_priv[tx] = 0;
  }
  __syncthreads(); //initialize private histogram
  //add to private histogram
  if (idx < size) {
    atomicAdd( &(histogram_priv[grayimg[idx]]), 1);

  }
  __syncthreads(); //once this block is done with its private histogram, contribute to global histogram.

  if (tx < HISTOGRAM_LENGTH){
    atomicAdd( &(histogram[tx]), histogram_priv[tx]);
  }
  //printf("Block: %d; histogram[tx]: %d\n", blockIdx.x, histogram[tx]);
}

__global__ void cdf_kernel ( float* out, unsigned int *histogram, int length, unsigned int imgSize) {

  __shared__ float cdf[HISTOGRAM_LENGTH];

  unsigned int tx = threadIdx.x;

  int firstHalfIdx = tx;
  int secondHalfIdx = blockDim.x + tx;
  cdf[firstHalfIdx] = firstHalfIdx < length ? (float) histogram[firstHalfIdx]/imgSize : 0.0f; //should always evaluate to true
  cdf[secondHalfIdx] = secondHalfIdx < length ? (float) histogram[secondHalfIdx]/imgSize : 0.0f; //always true
  __syncthreads();
  //take care of power of 2 indices
  for (int stride = 1; stride <= blockDim.x; stride *= 2){
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if (index < length){
      cdf[index] += cdf[index - stride];
    }
    __syncthreads();
  }
  //reverse reduction
  for (int stride = blockDim.x/2; stride > 0; stride /= 2){
    int index = (threadIdx.x + 1)*stride*2-1;
    if (index + stride < length){
      cdf[index + stride] += cdf[index];
    }
    __syncthreads();
  }
  //after cdf has the correct values, just copy it over to histogram
  if (firstHalfIdx < length)
    out[firstHalfIdx] = cdf[firstHalfIdx];
  if (secondHalfIdx < length){
    out[secondHalfIdx] = cdf[secondHalfIdx];
  }

  //printf("cdf[%d]: %f\n", firstHalfIdx, out[firstHalfIdx]);
  //printf("cdf[%d]: %f\n", secondHalfIdx, out[secondHalfIdx]);
}

__device__ unsigned char correct_color(float *cdf, unsigned char val, float minimum){
  float t = max(255.0*(cdf[val] - minimum)/(1 - minimum),0.0f);
  return (unsigned char) min(t, 255.0f);
}

__global__ void equalize_kernel ( unsigned char *uCharImage, float *cdf, int length) {
  int tx = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tx;
  float minimum = cdf[0];

  if ( idx < length ){
    uCharImage[idx] = correct_color(cdf, uCharImage[idx], minimum);

  }
  //if (blockIdx.x == 10){
  //  printf("blockidx: %d; idx: %d; tx: %d; uCharImage[idx]: %d\n", blockIdx.x, idx, threadIdx.x, uCharImage[idx]);
  //}
}

__global__ void castFloat_kernel (float *out, unsigned char *in, unsigned int size){
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx < size ){
    out[idx] = (float) (in[idx]/255.0);
  }
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
  unsigned int * histogram;
  float * cdf;

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
  cudaMalloc((void **) &deviceGrayImageData, imageWidth * imageHeight * sizeof(unsigned char)); //no channels needed
  cudaMalloc((void **) &histogram, imageWidth * imageHeight * sizeof(unsigned int)); //no channels needed
  cudaMalloc((void **) &cdf, HISTOGRAM_LENGTH * sizeof(float));
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
  dim3 histGrid( (imageWidth * imageHeight - 1)/HISTOGRAM_LENGTH + 1, 1);
  dim3 histBlock ( HISTOGRAM_LENGTH, 1);
  histogram_kernel<<<histGrid, histBlock>>> ( histogram, deviceGrayImageData, imageHeight*imageWidth );


  //this part is probably better done on host....
  //compute cdf (256 wide -- prefix sum)
  dim3 cdfGrid( 1, 1, 1); //it is so small can just do 1
  dim3 cdfBlock( HISTOGRAM_LENGTH/2,1,1 );
  cdf_kernel<<<cdfGrid, cdfBlock>>> ( cdf, histogram, HISTOGRAM_LENGTH, imageHeight*imageWidth ); //just write the cdf into histogram in-place as we don't need it after this.

  //this part is also probably better done on host.... good for practice though
  //reduce cdf (find min, 256 wide)
  //dim3 minGrid( 1, 1, 1);
  //dim3 minBlock( HISTOGRAM_LENGTH/2,1,1 );
  //min_kernel<<<minGrid,minBlock>>>( min, histogram );
  //minimum = cdf[0]; //the first element of a cdf must be the minimum....

  //apply histogram equalization on ucharimage (image sized)
  dim3 correctGrid( (imageWidth * imageHeight * imageChannels - 1)/LINEAR_WIDTH + 1, 1, 1);
  dim3 correctBlock ( LINEAR_WIDTH, 1, 1); //just use linear stuff for one-to-one mapping functions.
  equalize_kernel<<<correctGrid, correctBlock>>> ( deviceUcharImageData, cdf, imageHeight*imageWidth*imageChannels);

  //cast back to floating point image (image sized)
  dim3 floatGrid( (imageWidth * imageHeight * imageChannels - 1)/LINEAR_WIDTH + 1, 1, 1);
  dim3 floatBlock ( LINEAR_WIDTH, 1, 1);
  castFloat_kernel<<<floatGrid,floatBlock>>> (deviceOutputImageData, deviceUcharImageData, imageHeight*imageWidth*imageChannels);

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData,
      deviceOutputImageData,
      imageWidth * imageHeight * imageChannels * sizeof(float),
      cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");


  wbSolution(args, outputImage);

  cudaFree(deviceOutputImageData);
  cudaFree(deviceInputImageData);
  cudaFree(deviceUcharImageData);
  cudaFree(deviceGrayImageData);
  cudaFree(histogram);
  cudaFree(cdf);

  wbImage_delete(inputImage);
  wbImage_delete(outputImage);

  return 0;
}


