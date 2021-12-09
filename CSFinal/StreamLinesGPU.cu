//
// Created by Manuel Santana on 11/15/2021.
//

#include <cstdio>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>


//A structure to make the passing of constants easier.
struct constants{
	int NUM_ROWS;
	int NUM_COLUMNS;
	int VAL_PER_CELL;
	int THREADS_PER_BLOCK;
};

typedef struct constants Constants;


//Device functions 
__device__ float biLinInter(float* globalVf, float startX, float startY, bool isX, Constants Cs){
    /*
     * Input:
     *      grid: The matrix of streamline values.
     *      startPointX: The x coordinate of the point to be interpolated around
     *      startPointY: The y coordinate of the point to be interpolated around
     *      isX: if true interpolate the x value, if false interpolate the y value.
     * Output:
     *      The x or y value at the point.
     */
    //Off set
    int offset;
    if (isX){
        offset = 0;
    }else{
        offset = 1;
    }
    //X goes from left to right, y goes bottom to top.
    int x_1 = floor(startX); int x_2 = ceil(startX);
    int y_1 = floor(startY); int y_2 = ceil(startY);

    // Calculate the corresponding indices in the grid for the values.
    int x_1Loc = x_1 * Cs.VAL_PER_CELL; int x_2Loc = x_2 * Cs.VAL_PER_CELL;
    int y_1Loc = y_1 * Cs.NUM_COLUMNS * Cs.VAL_PER_CELL; int y_2Loc = y_2 * Cs.NUM_COLUMNS * Cs.VAL_PER_CELL;
    

    return  globalVf[y_1Loc + x_1Loc + offset] * (x_2 - startX) * (y_2 - startY) +
	    globalVf[y_1Loc + x_2Loc + offset] * (startX - x_1) * (y_2 - startY) +
	    globalVf[y_2Loc + x_1Loc + offset] * (x_2 - startX) * (startY - y_1) +
	    globalVf[y_2Loc + x_2Loc + offset] * (startX - x_1) * (startY - y_1);
    
}

__device__
float linearInter(float* globalVf, float startX, float startY, bool isX, bool isXAxis, Constants Cs){
    /*
     * Function to do a 1 directional linear interpolation
     * Inputs:
     *       grid: Array containing the vector fields.
     *       startX: The starting X value.
     *       startY: The starting Y value.
     *       isX   : True if X Value is needed, False if Y value is needed.
     *       isXAxis: True if the interpolation is in the x direction, false if in the Y direction.
     */
    // The offset gets the x or y value from the grid.
    int offset;
    if (isX){
        offset = 0;
    }else{
        offset = 1;
    }
    if (isXAxis){
        int x_0 = floor(startX);
        int x_1 = ceil(startX);
        //We assume that the y value is the same for the linear interpolation
        int yCoord = floor(startY) * Cs.NUM_COLUMNS * Cs.VAL_PER_CELL;
        float f_0 = globalVf[yCoord + x_0 * Cs.VAL_PER_CELL + offset];
        float f_1 = globalVf[yCoord + x_1 * Cs.VAL_PER_CELL + offset];
     	return f_0 + (startX - x_0) * (f_1 - f_0); 
    }else{
        int y_0 = floor(startY);
        int y_1 = ceil(startY);
        //We assume that the x coordinate is the same both times.
        int xCoord = floor(startX) * Cs.VAL_PER_CELL;
        float f_0 = globalVf[y_0 * Cs.NUM_COLUMNS * Cs.VAL_PER_CELL + xCoord + offset];
        float f_1 = globalVf[y_1 * Cs.NUM_COLUMNS * Cs.VAL_PER_CELL + xCoord + offset]; 
        return f_0 + (startY - y_0) * (f_1 - f_0);
    }
}

__device__
float funcValue(float* globalVf, float startX, float startY, bool isX, Constants Cs) {
    /*
     * Function to return an interpolated value from the vector field
     * Inputs:
     *        globalVf: The array containing the vector field.
     *        startX: The starting value of x.
     *        startY: The starting value of y.
     *        isX   : True if trying to get the x value at that point. False if getting the Y value.
     * Output:
     *        Interpolated function value.
     */
    //No interpolation needed case.
    if ((ceil(startX) == floor(startX)) && (ceil(startY) == floor(startY))) {
        int offset;
        if (isX) {
            offset = 0;
        } else {
            offset = 1;
        }
        int xVal = (int) startX;//Get the integer form for indexing.
        int yVal = (int) startY;
        return globalVf[yVal * Cs.NUM_COLUMNS * Cs.VAL_PER_CELL + xVal * Cs.VAL_PER_CELL + offset]; 
    }//Case when we only need linear interpolation along the y axis
    else if ((ceil(startX) == floor(startX)) && (ceil(startY) != floor(startY))) {
        //False here shows that it is the y axis.
        return linearInter(globalVf, startX, startY, isX, false,Cs);
    }//Case when we only need linear interpolation along the x axis
    else if ((ceil(startX) != floor(startX)) && (ceil(startY) == floor(startY))) {
        //True here shows we go along the x axis.
        return linearInter(globalVf, startX, startY, isX, true,Cs);
    }//Case when we need bilinear interpolation
    else {
        return biLinInter(globalVf, startX, startY, isX, Cs);
    }
}

__device__
bool isNotInBounds(float x, float y, Constants Cs){
    /*
     * Makes sure the x and y values stay within bounds.
     */
    return (x < 0) || (x > Cs.NUM_COLUMNS - 1) || (y < 0) || (y > Cs.NUM_ROWS - 1);

}


__device__
void rk4(float* globalVf, float startX,float startY, float dt, float* newX, float* newY, Constants Cs){
    /* Returns a new coordinate in the given directions by using the Runge Kutta 4th order method assuming bilinear interpolation
     * is needed.
     * Inputs:
     *        grid: The vector field of directions.
     *        startX: The current estimated value of the function.
     *        startY: The current starting y value.
     *        dt: The change in the time step.
     *        newX: pointer to the value where the new x should be stored.
     *        newY: pointer to the value where the new y should be stored.
     */
    //Booleans to get the correct values of x and y from the bilinear interpolation
     bool isX = true;
     bool isY = false;
     //k's represent x points and l's represent y points. See this document for an explanation.
     //https://wps.prenhall.com/wps/media/objects/884/905485/chapt4/proj4.3A/proj4-3A.pdf
     float k1 = funcValue(globalVf,startX,startY,isX,Cs);
     float l1 = funcValue(globalVf,startX,startY,isY,Cs);
     float x1 = startX + dt * k1 / 2.0;
     float y1 = startY + dt * l1 /2.0;
     //If x1 or y1 is out of bounds, we assume the streamline left the grid and stop it there.
     if(isNotInBounds(x1,y1,Cs)){return;}
     float k2 = funcValue(globalVf,x1,y1,isX,Cs);
     float l2 = funcValue(globalVf,x1,y1,isY,Cs);
     float x2 = startX + dt * k2 / 2.0;
     float y2 = startY + dt * l2 / 2.0;
     if(isNotInBounds(x2,y2,Cs)){return;}
     float k3 = funcValue(globalVf,x2,y2,isX,Cs);
     float l3 = funcValue(globalVf,x2,y2,isY,Cs);
     float x3 = startX + dt * k3;
     float y3 = startY + dt * l3;
     if(isNotInBounds(x3,y3,Cs)){return;}
     float k4 = funcValue(globalVf,x3,y3,isX,Cs);
     float l4 = funcValue(globalVf,x3,y3,isY,Cs);
     float kTotal = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
     float lTotal = (1.0 / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4);
     //Change the total value.
     *newX = startX + kTotal * dt;
     *newY = startY + lTotal * dt;
}

//Kernels
__global__
void computeStreamLines(float* globalVf, float* streamLines, int numberOfIterations, float dt, Constants Cs){
    /*
     * Kernals to compute the vector streamlines.
     * Inputs:
     *       globalVectorField : 1-dimensional array of the vector field.
     *       streamLines       : Where to write the data from the computed streamlines.
     *       numberOfIterations: How many iterations to run.
     *       dt                : the time step size.
     */   
 
    //The row to read things from global memory, also the start row.
    int globalRow = blockDim.y * blockIdx.y + threadIdx.y;
    int globalRowStart = globalRow * numberOfIterations * Cs.VAL_PER_CELL; 
    
   //This need to loop the rows that we have.
   //This loop is replaced by the threads.
   // for (int i = 0; i < NUM_ROWS; i++){
   float  currX = 0;
   float  currY = globalRow; 
   float  newX = 0;
   float  newY = 0;
   for (int j = 0; j < numberOfIterations; j++) {
            //Check boundary case. If it leaves the grid then we consider it done.
       if (isNotInBounds(currX, currY,Cs)) {
           streamLines[globalRow * numberOfIterations * Cs.VAL_PER_CELL + j * Cs.VAL_PER_CELL] = currX;
           streamLines[globalRow * numberOfIterations * Cs.VAL_PER_CELL + j * Cs.VAL_PER_CELL + 1] = currY;
       }else{
           rk4(globalVf, currX, currY, dt, &newX, &newY,Cs);
           streamLines[globalRowStart + j * Cs.VAL_PER_CELL] = newX;
           //Here we have an offset of 1 for the y value.
           streamLines[globalRowStart + j * Cs.VAL_PER_CELL + 1] = newY;
           currX = newX;
           currY = newY;
       }
   }
}


//Function to make sure CUDA functions go properly.
cudaError_t checkCuda(cudaError_t result){
	if (result != cudaSuccess){
		fprintf(stderr,"CUDA Runtime Error. %s\n",cudaGetErrorString(result));
	}
	return result;
}


int main(){
     
      
     //Set up timers. _t is for the total time, and _k is for the kernal.
     cudaEvent_t start_t, finish_t, start_k, finish_k; 	
     cudaEventCreate(&start_t);
     cudaEventCreate(&finish_t);
     cudaEventCreate(&start_k);
     cudaEventCreate(&finish_k);

     cudaEventRecord(start_t);

     Constants Cs;

     //Constant values needed for the functions. 
     Cs.NUM_ROWS = 600;
     Cs.NUM_COLUMNS = 1300;
     Cs.VAL_PER_CELL = 2;
     //50 Is a good number
     Cs.THREADS_PER_BLOCK = 50;  

    FILE* vectorDoc;
    vectorDoc = fopen("cyl2d_1300x600_float32[2].raw","rb");
    if (vectorDoc == NULL){
        std::cout << "Read File didn't open" << std::endl;
    }
    float* vectorField_h = new float[Cs.NUM_ROWS * Cs.NUM_COLUMNS * Cs.VAL_PER_CELL];
    fread(vectorField_h,Cs.VAL_PER_CELL * sizeof(float),Cs.NUM_COLUMNS * Cs.NUM_ROWS,vectorDoc);
    fclose(vectorDoc);
   //The next two parameters can be varied, but dt = 0.1 and totalTime = 1200 seems to give a complete picture.
   double dt = 0.1;
   float totalTime = 1200;
   int numberOfIterations = totalTime / dt;
  
   //Set up block sizes, and allocate memory on Cuda. 
   int numBlocks = Cs.NUM_ROWS / Cs.THREADS_PER_BLOCK; 
   dim3 dimGrid(1,numBlocks,1); 
   dim3 dimBlock(1,Cs.THREADS_PER_BLOCK,1);
 	  
   float* vectorField_d; float* streamLines_d;
   
   //An array where each streamline is held. We will make each streamline have the same number of values.
   float* streamLines_h = new float[Cs.NUM_ROWS * numberOfIterations * Cs.VAL_PER_CELL];
    
   //Allocated the memory on the device.   
   checkCuda(cudaMalloc(&vectorField_d, Cs.NUM_ROWS * Cs.NUM_COLUMNS * Cs.VAL_PER_CELL * sizeof(float)));
   checkCuda(cudaMalloc(&streamLines_d, Cs.NUM_ROWS * numberOfIterations * Cs.VAL_PER_CELL * sizeof(float)));
   
   checkCuda(cudaMemcpy(vectorField_d,vectorField_h, Cs.NUM_ROWS * Cs.NUM_COLUMNS * Cs.VAL_PER_CELL * sizeof(float),cudaMemcpyHostToDevice));
   
   cudaEventRecord(start_k);
   computeStreamLines<<<dimGrid,dimBlock>>>(vectorField_d,streamLines_d,numberOfIterations,dt,Cs);
   cudaEventRecord(finish_k);   

   cudaEventSynchronize(finish_k);

   //Copy the streamlines back from the GPU.
   checkCuda(cudaMemcpy(streamLines_h,streamLines_d,Cs.NUM_ROWS * numberOfIterations * Cs.VAL_PER_CELL * sizeof(float),cudaMemcpyDeviceToHost)); 	
    
   
   checkCuda(cudaFree(vectorField_d));
   checkCuda(cudaFree(streamLines_d)); 
   delete[] vectorField_h;

   //Write to a file:
   FILE* StreamLineCSV;
   StreamLineCSV = fopen("GPUStreamLines.csv","w+");
   fprintf(StreamLineCSV, "Line ID, X_Coordinate, Y_Coordinate\n");
   for (int streamlineID = 0; streamlineID < Cs.NUM_ROWS;streamlineID++){
       for (int k = 0; k < numberOfIterations;k++){
           int xCoord = streamlineID * numberOfIterations * Cs.VAL_PER_CELL + k * Cs.VAL_PER_CELL;
           int yCoord = xCoord + 1;
           fprintf(StreamLineCSV,"%d, %f, %f\n",streamlineID,streamLines_h[xCoord],streamLines_h[yCoord]);
       }
   }

   fclose(StreamLineCSV);
   delete[] streamLines_h;
   cudaEventRecord(finish_t);
   float totalTimeElapsed, kernelTime;
   cudaEventElapsedTime(&totalTimeElapsed,start_t,finish_t);
   cudaEventElapsedTime(&kernelTime,start_k,finish_k);
   std::cout<< "The total time was " << totalTimeElapsed << " and the stream line computation time was " << kernelTime << std::endl;
   return 0;
}
