//
// Created by Manuel Santana on 12/1/2021.
//

#include <cstdio>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

//Global variables specific to our vector field.
int NUM_ROWS = 600;
int NUM_COLUMNS = 1300;
int VAL_PER_CELL = 2;
//CACHE_TAG is the message tag for sending a cache, and DONE_TAG is for sending you are done.
int CACHE_TAG = 4;
int DONE_TAG = 7;


bool isInRange(int yVal, int yLow, int yHigh){
	//A simple function to determing if a single value is in some block.
	return (yVal >= yLow) && (yVal <= yHigh);
}

void getCache(float* cache,float* localVectorField, int* cacheRank, int neededY, int localRowCount){
        /*
 	* Function to request the new cache if needed. Note that since we are distributing from rows 
 	* the node only needs to ask for a new cache if the y value is out of bounds. 
 	* Inputs:
 	* 	localVectorField : The local vector field data.
 	*       neededY: The Y value needed to determing which node to request from.
 	*       localRowCount: How many rows does each node have to calculate which to receive from. 
 	* OutPuts:
 	*       cache: A pointer to the new cache. 
 	*       cacheRank: The new rank of the node from which the cache came from. 
	*/
         
 	int nodeToMessage = neededY / localRowCount;
        int message = 1;
	int blockSize = localRowCount * NUM_COLUMNS * VAL_PER_CELL;
 
       //Have the process send for the new cache, while still checking to see if it needs to send its cache to other processes.
       //This becomes especially important when computing with a large number of processes so there is more communication going on.
        MPI_Request sendRequest;
	MPI_Isend(&message,1,MPI_INT,nodeToMessage,CACHE_TAG,MPI_COMM_WORLD,&sendRequest);
	MPI_Request recvRequest;
	MPI_Status recvStatus;
	MPI_Irecv(cache,blockSize,MPI_FLOAT,nodeToMessage,CACHE_TAG,MPI_COMM_WORLD,&recvRequest);	
        int cacheFlag = 0;
	int otherMessage = 0;
	int recvFlag = 0;
        MPI_Status cacheStatus;   
	while (recvFlag = 0){
	      MPI_Test(&recvRequest, &recvFlag, &recvStatus);
	      MPI_Iprobe(MPI_ANY_SOURCE,CACHE_TAG,MPI_COMM_WORLD,&cacheFlag,&cacheStatus);
	      if (cacheFlag != 0){
                 int localDataCount = localRowCount * NUM_COLUMNS * VAL_PER_CELL;
                 MPI_Sendrecv(localVectorField,localDataCount,MPI_FLOAT,cacheStatus.MPI_SOURCE,CACHE_TAG,&otherMessage,1,MPI_INT,MPI_ANY_SOURCE,CACHE_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		 cacheFlag = 0;
		 MPI_Status newStatus;
		 cacheStatus = newStatus;
	      }
	}

	//update the current cache rank after the message has been succesfully recieved.
        *cacheRank = nodeToMessage;
}


float getGridValue(float* localVf,float* cacheVf,int myRank, int* cacheRank, int x, int y, int localRowCount, int offset){
/* A function to get the value of the vector field grid, depending on the neeeded x and y value.
 * Inputs:
 *       localVf: The local vector field data.
 *       cacheVf: The local cache value.
 *       myRank: The current nodes rank.
 *       cacheRank: The current rank of node from which the current cache came from. If -1 then no cache yet.
 *       x,y: The value needed.
 *       localRowCount: How many rows each node has
 *       offset: Value to determing if we are getting the x or y value from the grid.
 * Outputs:
 *        returnValue: The needed value from the grid.
 *        cacheVf: The updated cache if an updated cache was needed.
 *        cacheRank: The updated cache rank if an updated cache was needed.
 */
     
    int yLow = myRank * localRowCount; int yHigh = (myRank + 1) * localRowCount - 1; 
    int yLow_c = (*cacheRank) * localRowCount; int yHigh_c = ((*cacheRank) + 1) * localRowCount - 1; 
    
    int xLoc = x * VAL_PER_CELL;    
    //The next location for the y values will be the same regardless if they are in the local or cached vf. 
    int yLoc = (y % localRowCount) * NUM_COLUMNS * VAL_PER_CELL; 
 
    float returnValue = 0;
    //Cases to see if we need to use the cache or not.
    if (isInRange(y,yLow,yHigh)){
          returnValue = localVf[yLoc + xLoc + offset];
    }else if (isInRange(y,yLow_c,yHigh_c)){
          returnValue = cacheVf[yLoc + xLoc + offset]; 
    }else{
       getCache(cacheVf,localVf, cacheRank, y, localRowCount);  
       returnValue = cacheVf[yLoc + xLoc + offset];
    }
    return returnValue;
}

float biLinInter(float* localVf, float* cacheVf, int myRank, int* cacheRank, int localRowCount, float startX, float startY, bool isX){
    /*
     * Input:
     *      localVf: Local streamline values.
     *      cacheVf: Current cached streamline values. 
     *      myRank : Rank of the current node.
     *      cacheRank: The rank of the node where the cache was gotten from. Passed as a pointer so it can change if the cache needs to be changed.
     *      startPointX: The x coordinate of the point to be interpolated around
     *      startPointY: The y coordinate of the point to be interpolated around
     *      isX: if true interpolate the x value, if false interpolate the y value.
     *      localRowCount: The number of rows per a node.
     * Output:
     *      The x or y value at the point.
     */
    
    //This offsets gets the x or y value.
    int offset;
    if (isX){
        offset = 0;
    }else{
        offset = 1;
    }
    //X goes from left to right, y goes bottom to top.
    int x_1 = floor(startX); int x_2 = ceil(startX);
    int y_1 = floor(startY); int y_2 = ceil(startY);
 
    float returnValue = getGridValue(localVf,cacheVf,myRank,cacheRank,x_1,y_1,localRowCount,offset) * (x_2 - startX) * (y_2 - startY) +
                        getGridValue(localVf,cacheVf,myRank,cacheRank,x_2,y_1,localRowCount,offset) * (startX - x_1) * (y_2 - startY) +
                        getGridValue(localVf,cacheVf,myRank,cacheRank, x_1,y_2,localRowCount,offset) * (x_2 - startX) * (startY - y_1) +
                        getGridValue(localVf,cacheVf,myRank,cacheRank,x_2,y_2,localRowCount,offset) * (startX - x_1) * (startY - y_1);
    return returnValue;
}

float linearInter(float* localVf, float* cacheVf,int myRank, int* cacheRank, int localRowCount, float startX, float startY, bool isX, bool isXAxis){
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
        int y = floor(startY);
        float f_0 = getGridValue(localVf,cacheVf,myRank,cacheRank,x_0,y,localRowCount,offset);    
        float f_1 = getGridValue(localVf,cacheVf,myRank,cacheRank,x_1,y,localRowCount,offset);
        return f_0 + (startX - x_0) * (f_1 - f_0);
    }else{
        int y_0 = floor(startY);
        int y_1 = ceil(startY);
        //We assume that the x coordinate is the same both times.
        int x = floor(startX);
        float f_0 = getGridValue(localVf,cacheVf,myRank,cacheRank,x,y_0,localRowCount,offset);    
        float f_1 = getGridValue(localVf,cacheVf,myRank,cacheRank,x,y_1,localRowCount,offset);
        return f_0 + (startY - y_0) * (f_1 - f_0);
    }
}

float funcValue(float* localVf, float* cacheVf,int myRank, int* cacheRank, int localRowCount, float startX, float startY, bool isX) {
    /*
     * Function to return an interpolated value from the vector field
     * Inputs:
     *        localVf: The array containing the local data vectorfield.
     *        cacheVf: The cached vector field data.
     *        myRank: The rank of the node.
     *        localRowCount: How many rows are needed.
     *        cacheRank : The rank of the process from the current cache came from. If no current cache the value is -1.
     *        startX: The starting value of x.
     *        startY: The starting value of y.
     *        isX   : True if trying to get the x value at that point. False if getting the Y value.
     * Output:
     *        Interpolated function value.
     *        cacheVf and cacheRank will be updated if needed.
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
                
        return getGridValue(localVf,cacheVf,myRank,cacheRank,xVal,yVal,localRowCount,offset);
    }//Case when we only need linear interpolation along the y axis
    else if ((ceil(startX) == floor(startX)) && (ceil(startY) != floor(startY))) {
        //False here shows that it is the y axis.
        return linearInter(localVf,cacheVf,myRank,cacheRank,localRowCount,startX, startY, isX, false);
    }//Case when we only need linear interpolation along the x axis
    else if ((ceil(startX) != floor(startX)) && (ceil(startY) == floor(startY))) {
        //True here shows we go along the x axis.
        return linearInter(localVf,cacheVf,myRank,cacheRank,localRowCount, startX, startY, isX, true);
    }//Case when we need bilinear interpolation
    else {
        return biLinInter(localVf,cacheVf,myRank,cacheRank, localRowCount, startX, startY, isX);
    }
}


bool isNotInBounds(float x, float y){
    /*
     * Returns true if x and y are within the bounds of the vector field..
     */
    return (x < 0) || (x > NUM_COLUMNS - 1) || (y < 0) || (y > NUM_ROWS - 1);
}


void rk4(float* localVf,float* cacheVf, int myRank, int* cacheRank, int localRowCount, float startX,float startY, float dt, float* newX, float* newY){
    /* Returns a new coordinate in the given directions by using the Runge Kutta 4th order method assuming bilinear interpolation
     * is needed.
     * Inputs:
     *        localVf: The array containing the local data vectorfield.
     *        cacheVf: The cached vector field data.
     *        myRank: The rank of the node.
     *        localRowCount: How many rows are needed.
     *        cacheRank : The rank of the process from the current cache came from. If no current cache the value is -1.
     *        startX: The current estimated value of the function.
     *        startY: The current starting y value.
     *        dt: The change in the time step.
     * Outputs:
     *        newX: pointer to the value where the new x should be stored.
     *        newY: pointer to the value where the new y should be stored.
     *        Note that cacheVf and cacheRank may be updated if needed.
     */

    //Booleans to get the correct values of x and y from the bilinear interpolation
 //   if (myRank == 0){std::cout << "Made it into RK4" << std::endl;}
    bool isX = true;
    bool isY = false;
    //k's represent x points and l's represent y points. See this document for an explanation.
    //https://wps.prenhall.com/wps/media/objects/884/905485/chapt4/proj4.3A/proj4-3A.pdf
    float k1 = funcValue(localVf,cacheVf,myRank,cacheRank,localRowCount,startX,startY,isX);
    float l1 = funcValue(localVf,cacheVf,myRank,cacheRank,localRowCount,startX,startY,isY);
    float x1 = startX + dt * k1 / 2.0;
    float y1 = startY + dt * l1 /2.0;
    //If x1 or y1 is out of bounds, we assume the streamline left the grid and stop it there.
    if(isNotInBounds(x1,y1)){return;}
    float k2 = funcValue(localVf,cacheVf,myRank,cacheRank,localRowCount,x1,y1,isX);
    float l2 = funcValue(localVf,cacheVf,myRank,cacheRank,localRowCount,x1,y1,isY);
    float x2 = startX + dt * k2 / 2.0;
    float y2 = startY + dt * l2 / 2.0;
    if(isNotInBounds(x2,y2)){return;}
    float k3 = funcValue(localVf,cacheVf,myRank,cacheRank,localRowCount,x2,y2,isX);
    float l3 = funcValue(localVf,cacheVf,myRank,cacheRank,localRowCount,x2,y2,isY);
    float x3 = startX + dt * k3;
    float y3 = startY + dt * l3;
    if(isNotInBounds(x3,y3)){return;}
    float k4 = funcValue(localVf,cacheVf,myRank,cacheRank,localRowCount,x3,y3,isX);
    float l4 = funcValue(localVf,cacheVf,myRank,cacheRank,localRowCount,x3,y3,isY);
    float kTotal = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    float lTotal = (1.0 / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4);
    //Change the total value.
    *newX = startX + kTotal * dt;
    *newY = startY + lTotal * dt;
}

int main(){ 
 
    MPI_Init(NULL,NULL);
    //Timing variables for the total time and the start to finish time
    double totalStart, totalFinish, streamStart, streamFinish;
    totalStart = MPI_Wtime();

    int myRank; int commSize;
    //cacheRank is -1 if you have nothing in the cache.
    int cacheRank = -1;

    MPI_Comm_size(MPI_COMM_WORLD,&commSize);
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
    
    //We are going to assume that the commSize divides the number of rows. 
    if (NUM_ROWS % commSize != 0){
       if(myRank == 0){std::cout << "commSize must divide the number of rows (600)" <<std::endl;}
          return 0;
    }
        
    //The next two parameters can be varied, but dt = 0.1 and totalTime = 1200 seems to give a complete picture.
    double dt = 0.1;
    float totalTime = 1200;
    int numberOfIterations = totalTime / dt;

    //How many rows and how much vector field data each process gets. 
    int localRowCount = (NUM_ROWS / commSize); 
    int localDataCount =  localRowCount * NUM_COLUMNS * VAL_PER_CELL;
     
    //Pointers to the global, local, and cached vectorfield data. 
    float* totalVectorField; 
    float* localVectorField = new float[localDataCount];
    //Each process gets one cache they can hold someone else's streamline information. 
    float* cacheVectorField = new float[localDataCount];   

 
    //Rank 0 reads in the vector field and distributes it to all other processes. 
    if (myRank == 0){
         FILE* vectorDoc;
         vectorDoc = fopen("cyl2d_1300x600_float32[2].raw","rb");
         if (vectorDoc == NULL){
             std::cout << "Read File didn't open" << std::endl;
         }
         totalVectorField = new float[NUM_ROWS * NUM_COLUMNS * VAL_PER_CELL];
         fread(totalVectorField,VAL_PER_CELL * sizeof(float),NUM_COLUMNS * NUM_ROWS,vectorDoc);
         fclose(vectorDoc);
	}	
    streamStart = MPI_Wtime(); 
    MPI_Scatter(totalVectorField,localDataCount,MPI_FLOAT,localVectorField,localDataCount,MPI_FLOAT,0,MPI_COMM_WORLD); 
    
    //Local Start Row is the starting y value for each process.
    int localStartRow = localRowCount * myRank; 
    int localStreamCount = localRowCount * numberOfIterations * VAL_PER_CELL;
    //An array where each streamline is held. We will make each streamline have the same number of values.
    float* localStreamLines = new float[localStreamCount];
 

    //Have the process probe to see if anyone needs their cache. The message is unimportant, only that they recieved a message.
    int cacheFlag = 0; //flag to know if a message about caches has been received. 
    int recvMes = 0; 
    MPI_Status cacheStatus; //status to check if a message about caches has been received. 
    //Loop over the rows and compute the streamLines.
 for (int i = 0; i < localRowCount; i++) {
       float  currX = 0;
       float  currY = i + localStartRow;
       float  newX = 0;
       float  newY = 0;
       for (int j = 0; j < numberOfIterations; j++) {
            //Check for message data and attempt to send it.
	    MPI_Iprobe(MPI_ANY_SOURCE,CACHE_TAG,MPI_COMM_WORLD,&cacheFlag,&cacheStatus);
            if (cacheFlag != 0){
		//Recieve the message and send the data asked for, the probe for messages again.	
                MPI_Sendrecv(localVectorField,localDataCount,MPI_FLOAT,cacheStatus.MPI_SOURCE,CACHE_TAG,&recvMes,1,MPI_INT,MPI_ANY_SOURCE,CACHE_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Status newStatus;
		cacheStatus = newStatus;
                cacheFlag = 0;
            }
            	
            //Check boundary case. If it leaves the grid then we consider it done.
            if (isNotInBounds(currX, currY)) {
                localStreamLines[i * numberOfIterations * VAL_PER_CELL + j * VAL_PER_CELL] = currX;
                localStreamLines[i * numberOfIterations * VAL_PER_CELL + j * VAL_PER_CELL + 1] = currY;
            }else {
                //Recall that Cache Rank needs to be a pointer so it can be updated.
                rk4(localVectorField,cacheVectorField,myRank,&cacheRank,localRowCount, currX, currY, dt, &newX, &newY);	
                localStreamLines[i * numberOfIterations * VAL_PER_CELL + j * VAL_PER_CELL] = newX;
                //Here we have an offset of 1 for the y value.
                localStreamLines[i * numberOfIterations * VAL_PER_CELL + j * VAL_PER_CELL + 1] = newY;
                currX = newX;
                currY = newY;
            }
        }
    }

   
    //Variables to check if everyone is done. 
    bool allFinished = false; 
    MPI_Request doneRequest;//Request for asynchronous Bcast.  
    int doneMessage =0;
    if (myRank == 0){//Send gather when people are done, and still send data.
    	int doneFlag = 0;
    	MPI_Status doneStatus;
        int numThreadsFinished = 1;//This is to count that process 0 is finished at this point.
        while(numThreadsFinished < commSize){ 
	    MPI_Iprobe(MPI_ANY_SOURCE,DONE_TAG,MPI_COMM_WORLD,&doneFlag,&doneStatus);
            if(doneFlag != 0){//Check to see if there is a done message.
           	 MPI_Recv(&recvMes,1,MPI_INT,doneStatus.MPI_SOURCE,DONE_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
           	 numThreadsFinished = numThreadsFinished + 1;
           	 MPI_Status newStatus;
       	         doneStatus = newStatus;
	         doneFlag = 0;
            }   
	    MPI_Iprobe(MPI_ANY_SOURCE,CACHE_TAG,MPI_COMM_WORLD,&cacheFlag,&cacheStatus);
            if (cacheFlag != 0){//Check to see if someone needs my localData.
                MPI_Sendrecv(localVectorField,localDataCount,MPI_FLOAT,cacheStatus.MPI_SOURCE,CACHE_TAG,&recvMes,1,MPI_INT,MPI_ANY_SOURCE,CACHE_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Status newStatus;
		cacheStatus = newStatus;
                cacheFlag = 0;

            } 
         }
        allFinished = true; 
        MPI_Ibcast(&allFinished,1,MPI_CXX_BOOL,0,MPI_COMM_WORLD,&doneRequest);
    }else{//wait until everyone is done, and still be avaliable to send data.
         MPI_Request doneRequest; 
         MPI_Isend(&doneMessage,1,MPI_INT,0,DONE_TAG,MPI_COMM_WORLD,&doneRequest);
	 MPI_Request bcastRequest; 
         MPI_Ibcast(&allFinished,1,MPI_CXX_BOOL,0,MPI_COMM_WORLD,&bcastRequest);
	 MPI_Status bcastStatus;
	 int doneFlag = 0;
         //Probe for messages to get my data.
         while(doneFlag == 0){
	    MPI_Iprobe(MPI_ANY_SOURCE,CACHE_TAG,MPI_COMM_WORLD,&cacheFlag,&cacheStatus);   
            if (cacheFlag != 0){//Check to see if someone needs my localData.
                MPI_Sendrecv(localVectorField,localDataCount,MPI_FLOAT,cacheStatus.MPI_SOURCE,CACHE_TAG,&recvMes,1,MPI_INT,MPI_ANY_SOURCE,CACHE_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Status newStatus;
		cacheStatus = newStatus;
                cacheFlag = 0;
            }  
            MPI_Test(&bcastRequest,&doneFlag,&bcastStatus);
         } 
    }    

    streamFinish = MPI_Wtime();
    if (myRank == 0){std::cout << "Total Stream Time is " << streamFinish - streamStart << std::endl;}

    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] localVectorField;
    delete[] cacheVectorField;
    //Write to a file:
    float* totalStreamLines;// = new float[ NUM_ROWS * numberOfIterations * VAL_PER_CELL];
    if (myRank == 0){totalStreamLines = new float[NUM_ROWS * numberOfIterations * VAL_PER_CELL];}
    MPI_Gather(localStreamLines,localStreamCount,MPI_FLOAT,totalStreamLines,localStreamCount,MPI_FLOAT,0,MPI_COMM_WORLD);
    if (myRank == 0){
        //totalStreamLines = new float[NUM_ROWS * numberOfIterations * VAL_PER_CELL]; 
       // MPI_Gather(localStreamLines,localStreamCount,MPI_FLOAT,totalStreamLines,localStreamCount,MPI_FLOAT,0,MPI_COMM_WORLD);
    	FILE* StreamLineCSV;
    	StreamLineCSV = fopen("MPILines.csv","w+");
    	fprintf(StreamLineCSV, "Line ID, X_Coordinate, Y_Coordinate\n");
    	for (int streamlineID = 0; streamlineID < NUM_ROWS;streamlineID++){
	        for (int k = 0; k < numberOfIterations;k++){
            		int xCoord = streamlineID * numberOfIterations * VAL_PER_CELL + k * VAL_PER_CELL;
            		int yCoord = xCoord + 1;
            		fprintf(StreamLineCSV,"%d, %f, %f\n",streamlineID,totalStreamLines[xCoord],totalStreamLines[yCoord]);
        		}
    		}

        fclose(StreamLineCSV);
     }
     

    totalFinish = MPI_Wtime();
    if(myRank == 0){std::cout << "The total time was " << totalFinish - totalStart << std::endl;}
    MPI_Finalize();
    return MPI_SUCCESS;
}

