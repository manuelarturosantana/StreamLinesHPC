//
// Created by Manuel Santana on 11/15/2021.
//

#include <cstdio>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

//Global variables specific to our vector field.
int NUM_ROWS = 600;
int NUM_COLUMNS = 1300;
int VAL_PER_CELL = 2;

float biLinInter(float* grid, float startX, float startY, bool isX){
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
    int x_1Loc = x_1 * VAL_PER_CELL; int x_2Loc = x_2 * VAL_PER_CELL;
    int y_1Loc = y_1 * NUM_COLUMNS * VAL_PER_CELL; int y_2Loc = y_2 * NUM_COLUMNS * VAL_PER_CELL;
    float returnValue = grid[y_1Loc + x_1Loc + offset] * (x_2 - startX) * (y_2 - startY) +
                        grid[y_1Loc + x_2Loc + offset] * (startX - x_1) * (y_2 - startY) +
                        grid[y_2Loc + x_1Loc + offset] * (x_2 - startX) * (startY - y_1) +
                        grid[y_2Loc + x_2Loc + offset] * (startX - x_1) * (startY - y_1);

    return returnValue;
}

float linearInter(float* grid, float startX, float startY, bool isX, bool isXAxis){
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
        int yCoord = floor(startY) * NUM_COLUMNS * VAL_PER_CELL;
        float f_0 = grid[yCoord + x_0 * VAL_PER_CELL + offset];
        float f_1 = grid[yCoord + x_1 * VAL_PER_CELL + offset];
        return f_0 + (startX - x_0) * (f_1 - f_0);
    }else{
        int y_0 = floor(startY);
        int y_1 = ceil(startY);
        //We assume that the x coordinate is the same both times.
        int xCoord = floor(startX) * VAL_PER_CELL;
        //+1 is to get the y coordinate.
        float f_0 = grid[y_0 * NUM_COLUMNS * VAL_PER_CELL + xCoord + offset];
        float f_1 = grid[y_1 * NUM_COLUMNS * VAL_PER_CELL + xCoord + offset];
        return f_0 + (startY - y_0) * (f_1 - f_0);
    }
}

float funcValue(float* grid, float startX, float startY, bool isX) {
    /*
     * Function to return an interpolated value from the vector field
     * Inputs:
     *        grid: The array containing the vector field.
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
        return grid[yVal * NUM_COLUMNS * VAL_PER_CELL + xVal * VAL_PER_CELL + offset];
    }//Case when we only need linear interpolation along the y axis
    else if ((ceil(startX) == floor(startX)) && (ceil(startY) != floor(startY))) {
        //False here shows that it is the y axis.
        return linearInter(grid, startX, startY, isX, false);
    }//Case when we only need linear interpolation along the x axis
    else if ((ceil(startX) != floor(startX)) && (ceil(startY) == floor(startY))) {
        //True here shows we go along the x axis.
        return linearInter(grid, startX, startY, isX, true);
    }//Case when we need bilinear interpolation
    else {
        return biLinInter(grid, startX, startY, isX);
    }
}

bool isNotInBounds(float x, float y){
    /*
     * Makes sure the x and y values stay within bounds.
     */
    return (x < 0) || (x > NUM_COLUMNS - 1) || (y < 0) || (y > NUM_ROWS - 1);
}

void rk4(float* grid, float startX,float startY, float dt, float* newX, float* newY){
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
     float k1 = funcValue(grid,startX,startY,isX);
     float l1 = funcValue(grid,startX,startY,isY);
     float x1 = startX + dt * k1 / 2.0;
     float y1 = startY + dt * l1 /2.0;
     //If x1 or y1 is out of bounds, we assume the streamline left the grid and stop it there.
     if(isNotInBounds(x1,y1)){return;}
     float k2 = funcValue(grid,x1,y1,isX);
     float l2 = funcValue(grid,x1,y1,isY);
     float x2 = startX + dt * k2 / 2.0;
     float y2 = startY + dt * l2 / 2.0;
     if(isNotInBounds(x2,y2)){return;}
     float k3 = funcValue(grid,x2,y2,isX);
     float l3 = funcValue(grid,x2,y2,isY);
     float x3 = startX + dt * k3;
     float y3 = startY + dt * l3;
     if(isNotInBounds(x3,y3)){return;}
     float k4 = funcValue(grid,x3,y3,isX);
     float l4 = funcValue(grid,x3,y3,isY);
     float kTotal = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
     float lTotal = (1.0 / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4);
     //Change the total value.
     *newX = startX + kTotal * dt;
     *newY = startY + lTotal * dt;
}

int main(int argc, char* argv []){
    double totalTimeStart, totalTimeFinish, compTimeStart, compTimeFinish;
    totalTimeStart = omp_get_wtime();

    if (argc != 2){
        std::cout << "Incorrect Number of Input Aurguments" << std::endl;
        return -1;
    }
    int numThreads = atoi(argv[1]);

    FILE* vectorDoc;
    vectorDoc = fopen("cyl2d_1300x600_float32[2].raw","rb");
    if (vectorDoc == NULL){
        std::cout << "Read File didn't open" << std::endl;
    }
    float* vectorField = new float[NUM_ROWS * NUM_COLUMNS * VAL_PER_CELL];
    fread(vectorField,VAL_PER_CELL * sizeof(float),NUM_COLUMNS * NUM_ROWS,vectorDoc);
    fclose(vectorDoc);
   //The next two parameters can be varied, but dt = 0.1 and totalTime = 1200 seems to give a complete picture.
   double dt = 0.1;
   float totalTime = 1200;
   int numberOfIterations = totalTime / dt;
   //An array where each streamline is held. We will make each streamline have the same number of values.
    float* streamLines = new float[NUM_ROWS * numberOfIterations * VAL_PER_CELL];
    float currX; float currY; float newX; float newY;
    int i; int j;
    
    compTimeStart = omp_get_wtime();
#pragma omp parallel for num_threads(numThreads)  default(none) shared(dt,numberOfIterations,streamLines,vectorField,VAL_PER_CELL,NUM_ROWS)\
        private(i,j,currX,currY,newX,newY)
    for (i = 0; i < NUM_ROWS; i++) {
        currX = 0;
        currY = i;
        newX = 0;
        newY = 0;
        for (j = 0; j < numberOfIterations; j++) {
            //Check boundary case. If it leaves the grid then we consider it done.
            if (isNotInBounds(currX, currY)) {
                streamLines[i * numberOfIterations * VAL_PER_CELL + j * VAL_PER_CELL] = currX;
                streamLines[i * numberOfIterations * VAL_PER_CELL + j * VAL_PER_CELL + 1] = currY;
            }else{
                rk4(vectorField, currX, currY, dt, &newX, &newY);
                streamLines[i * numberOfIterations * VAL_PER_CELL + j * VAL_PER_CELL] = newX;
                //Here we have an offset of 1 for the y value.
                streamLines[i * numberOfIterations * VAL_PER_CELL + j * VAL_PER_CELL + 1] = newY;
                currX = newX;
                currY = newY;
            }
        }
    } 
    compTimeFinish = omp_get_wtime();

    //This code can be used to write the vector field to a readable csv file for exploration.
    //FILE* VectorFieldCSV;
    //VectorFieldCSV = fopen("VectorField.csv","w+");
    //fprintf(VectorFieldCSV, "Line ID, X_Coordinate, Y_Coordinate\n");
    //for (int rowID = 0; rowID < NUM_ROWS;rowID++){
    //    for (int l = 0; l < NUM_COLUMNS;l++){
    //        int xCoord = rowID * NUM_COLUMNS * VAL_PER_CELL + l * VAL_PER_CELL;
    //        int yCoord = xCoord + 1;
    //        fprintf(VectorFieldCSV,"%d, %lf, %lf\n",rowID,vectorField[xCoord],vectorField[yCoord]);
    //    }
   // }
   // fclose(VectorFieldCSV);
   //delete[] vectorField;

   //Write to a file:
   FILE* StreamLineCSV;
   StreamLineCSV = fopen("StreamLines.csv","w+");
   fprintf(StreamLineCSV, "Line ID, X_Coordinate, Y_Coordinate\n");
   for (int streamlineID = 0; streamlineID < NUM_ROWS;streamlineID++){
       for (int k = 0; k < numberOfIterations;k++){
           int xCoord = streamlineID * numberOfIterations * VAL_PER_CELL + k * VAL_PER_CELL;
           int yCoord = xCoord + 1;
           fprintf(StreamLineCSV,"%d, %f, %f\n",streamlineID,streamLines[xCoord],streamLines[yCoord]);
       }
   }

   fclose(StreamLineCSV);
   delete[] streamLines;
   totalTimeFinish = omp_get_wtime();
   double totalRunTime = totalTimeFinish - totalTimeStart;
   double totalCompTime = compTimeFinish - compTimeStart;
   std::cout << "The total run time was " << totalRunTime << " with computation time of " << totalCompTime << std::endl;

}
