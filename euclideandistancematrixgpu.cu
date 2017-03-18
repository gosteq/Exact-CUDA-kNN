#include "euclideandistancematrixgpu.h"
#include <cmath>
#include <float.h>
#include <set>
#include <algorithm>

#define TRUE 1
#define FALSE 0

#define OUT_OF_SOLUTION INT_MIN
#define IN_SOLUTION INT_MAX

#define NON_OF_BUCKET_IN_INTEREST 1000

struct DataPoint{
    int id;
    float distance;
};

bool compareByLengthMin(const DataPoint &a, const DataPoint &b){
    return a.distance < b.distance;
}

bool compareByLengthMax(const DataPoint &a, const DataPoint &b){
    return a.distance > b.distance;
}

bool compareByLengthMinCosine(const DataPoint &a, const DataPoint &b){
    return a.distance > b.distance;
}

bool compareByLengthMaxCosine(const DataPoint &a, const DataPoint &b){
    return a.distance < b.distance;
}

__global__
void kernelCosine(float* dataTable_device, size_t dataTable_Pitch, float* neighboursDistance_device, int* neighboursId_device,
                  float** resultLine, float** resultLine2, float** resultLine3, int** resultLineMarker, int* neighboursSize_device,
                  int dimensionOfEntity, int numberOfEntities, int numberOfNeighbors, int numberOfNeighborsWithMistake,
                  float mistakeShortage, float mistakeKindness, float minimalSizeOfBucket, int start, int end){
    __shared__ float biggestNumber[256];
    __shared__ float smalestNumber[256];
    __shared__ int hist[256];
    __shared__ float entity[256];
    __shared__ double minValue;
    __shared__ double maxValue;
    __shared__ double bias;
    __shared__ float lengthOfBucket;
    __shared__ int foundExactSolution;
    __shared__ int limitOfLengthOfBucketExceeded;
    __shared__ int alreadyFoundNumbers;
    __shared__ int interestingBucket;
    __shared__ int rewrittenNumbers;
    __shared__ int complement;

    int minimalNumbersToFind = float(numberOfNeighbors)*(1.0-mistakeShortage/100.0);
    int maximalNumbersToFind = float(numberOfNeighbors)*(1.0+mistakeKindness/100.0);

    int tid = threadIdx.x;
    int idOfBatch = blockIdx.x;
    int elementsPerBatch = ceil(float(end-start)/float(gridDim.x));
    int startOfTheBatch = elementsPerBatch*idOfBatch;
    int endOfTheBatch = elementsPerBatch*(idOfBatch+1) <= (end-start) ? elementsPerBatch*(idOfBatch+1) : (end-start);

    startOfTheBatch += start;
    endOfTheBatch += start;

    for(int i=startOfTheBatch ; i<endOfTheBatch ; ++i){
        biggestNumber[tid] = 0.0f;
        smalestNumber[tid] = FLT_MAX;

        for(int lp=tid ; lp<numberOfEntities ; lp+=blockDim.x){
            resultLine[idOfBatch][lp] = 0.0f;
            resultLine2[idOfBatch][lp] = 0.0f;
            resultLine3[idOfBatch][lp] = 0.0f;
        }
        __syncthreads();

        for(int d=0 ; d<dimensionOfEntity ; d+=256){
            if((tid < 256)&&(d+tid < dimensionOfEntity)){
                float* pElement = (float*)((char*)dataTable_device + (d+tid) * dataTable_Pitch) + i;
                entity[tid] = *pElement;
            }
            __syncthreads();

            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                float distanceAB = 0.0;
                float distanceA = 0.0;
                float distanceB = 0.0;
                for(int k=d ; (k<dimensionOfEntity)&&(k<(d+256)) ; ++k){
                    float* pElement = (float*)((char*)dataTable_device + k * dataTable_Pitch) + lp;
                    float pElementVal = *pElement;
                    distanceAB += entity[k-d]*pElementVal;
                    distanceA += entity[k-d]*entity[k-d];
                    distanceB += pElementVal*pElementVal;
                }
                resultLine[idOfBatch][lp] += distanceAB;
                resultLine2[idOfBatch][lp] += distanceA;
                resultLine3[idOfBatch][lp] += distanceB;
            }
            __syncthreads();
        }

        for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
            float distanceAB = resultLine[idOfBatch][lp];
            float distanceA = resultLine2[idOfBatch][lp];
            float distanceB = resultLine3[idOfBatch][lp];
            float distance = distanceAB/(sqrt(distanceA)*sqrt(distanceB));
            distance = (-1.0*distance)+1.0;
            resultLine[idOfBatch][lp] = distance;
            resultLineMarker[idOfBatch][lp] = 0;
            biggestNumber[tid] = max(biggestNumber[tid], ceil(distance));
            if(lp != i){
                smalestNumber[tid] = min(smalestNumber[tid], floor(distance));
            }
        }
        __syncthreads();
        if(tid == 0){
            resultLineMarker[idOfBatch][i] = OUT_OF_SOLUTION;
        }
        __syncthreads();

        if(tid < 32){
            for(int i=tid ; i<256 ; i+=32){
                biggestNumber[tid] = max(biggestNumber[tid], biggestNumber[i]);
                smalestNumber[tid] = min(smalestNumber[tid], smalestNumber[i]);
            }
        }
        if(tid == 0){
            #pragma unroll
            for(int c=0 ; c<32 ; ++c){
                biggestNumber[0] = max(biggestNumber[0], biggestNumber[c]);
                smalestNumber[0] = min(smalestNumber[0], smalestNumber[c]);
            }
        }
        __syncthreads();

        if(tid == 0){
            bias = smalestNumber[0];
            minValue = 0;
            maxValue = biggestNumber[0] - smalestNumber[0];
            maxValue = pow(2.0, ceil(log(maxValue+1.0)/log(2.0)));
            lengthOfBucket = (maxValue-minValue)/256.0;
            foundExactSolution = FALSE;
            limitOfLengthOfBucketExceeded = FALSE;
            alreadyFoundNumbers = 0;
            rewrittenNumbers = 0;
            complement = 0;
        }
        __syncthreads();

        while((foundExactSolution == FALSE) && (limitOfLengthOfBucketExceeded == FALSE)){
            hist[tid] = 0;
            interestingBucket = NON_OF_BUCKET_IN_INTEREST;
            __syncthreads();

            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                int marker = resultLineMarker[idOfBatch][lp];
                if(marker == 0){
                    int idOfBucketInHist = (resultLine[idOfBatch][lp]-bias-minValue)/lengthOfBucket;
                    atomicAdd(&hist[idOfBucketInHist], 1);
                    resultLineMarker[idOfBatch][lp] = idOfBucketInHist;
                }
            }
            __syncthreads();

            if(tid == 0){
                for(int k=1 ; k<256 ; ++k){
                    hist[k] += hist[k-1];
                }
            }
            __syncthreads();

            if((hist[tid]+alreadyFoundNumbers)>maximalNumbersToFind){
                atomicMin(&interestingBucket, tid);
            }

            if((tid == 0) && (alreadyFoundNumbers == maximalNumbersToFind) && (alreadyFoundNumbers == minimalNumbersToFind)){
                foundExactSolution = TRUE;
            }

            int tmpSum = hist[tid] + alreadyFoundNumbers;
            if((tmpSum>=minimalNumbersToFind) && (tmpSum<=maximalNumbersToFind)){
                foundExactSolution = TRUE;
            }

            if((tid == 0) && (lengthOfBucket < minimalSizeOfBucket)){
                limitOfLengthOfBucketExceeded = TRUE;
            }
            __syncthreads();

            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                int marker = resultLineMarker[idOfBatch][lp];
                if((marker < interestingBucket)&&(marker >= 0)){
                    resultLineMarker[idOfBatch][lp] = IN_SOLUTION;
                    atomicAdd(&alreadyFoundNumbers, 1);
                }else if((marker > interestingBucket)&&(marker < 256)){
                    resultLineMarker[idOfBatch][lp] = OUT_OF_SOLUTION;
                }else if(marker == interestingBucket){
                    resultLineMarker[idOfBatch][lp] = 0;
                }
            }

            if(tid == 0){
                bias = bias+interestingBucket*lengthOfBucket;
                minValue = 0.0;
                maxValue = lengthOfBucket;
                lengthOfBucket = (maxValue-minValue)/256.0;
            }
            __syncthreads();
        }
        __syncthreads();

        for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
            int marker = resultLineMarker[idOfBatch][lp];
            if(marker == IN_SOLUTION){
                int id = atomicAdd(&rewrittenNumbers, 1);
                neighboursDistance_device[i*numberOfNeighborsWithMistake+id] = -1.0*(resultLine[idOfBatch][lp]-1.0);
                neighboursId_device[i*numberOfNeighborsWithMistake+id] = lp;
            }
        }
        __syncthreads();

        if((limitOfLengthOfBucketExceeded == TRUE)&&(foundExactSolution == FALSE)){
            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                int marker = resultLineMarker[idOfBatch][lp];
                if(marker == 0){
                    int id2 = atomicAdd(&complement, 1);
                    if((id2+alreadyFoundNumbers) < minimalNumbersToFind){
                        int id = atomicAdd(&rewrittenNumbers, 1);
                        neighboursDistance_device[i*numberOfNeighborsWithMistake+id] = -1.0*(resultLine[idOfBatch][lp]-1.0);
                        neighboursId_device[i*numberOfNeighborsWithMistake+id] = lp;
                    }
                }
            }
        }
        __syncthreads();

        if(tid == 0){
            neighboursSize_device[i] = rewrittenNumbers;
        }
        __syncthreads();
    }
}

__global__
void kernelTaxicab(float* dataTable_device, size_t dataTable_Pitch, float* neighboursDistance_device, int* neighboursId_device,
                   float** resultLine, int** resultLineMarker, int* neighboursSize_device,
                   int dimensionOfEntity, int numberOfEntities, int numberOfNeighbors, int numberOfNeighborsWithMistake,
                   float mistakeShortage, float mistakeKindness, float minimalSizeOfBucket, int start, int end){
    __shared__ float biggestNumber[256];
    __shared__ float smalestNumber[256];
    __shared__ int hist[256];
    __shared__ float entity[256];
    __shared__ double minValue;
    __shared__ double maxValue;
    __shared__ double bias;
    __shared__ float lengthOfBucket;
    __shared__ int foundExactSolution;
    __shared__ int limitOfLengthOfBucketExceeded;
    __shared__ int alreadyFoundNumbers;
    __shared__ int interestingBucket;
    __shared__ int rewrittenNumbers;
    __shared__ int complement;

    int minimalNumbersToFind = float(numberOfNeighbors)*(1.0-mistakeShortage/100.0);
    int maximalNumbersToFind = float(numberOfNeighbors)*(1.0+mistakeKindness/100.0);

    int tid = threadIdx.x;
    int idOfBatch = blockIdx.x;
    int elementsPerBatch = ceil(float(end-start)/float(gridDim.x));
    int startOfTheBatch = elementsPerBatch*idOfBatch;
    int endOfTheBatch = elementsPerBatch*(idOfBatch+1) <= (end-start) ? elementsPerBatch*(idOfBatch+1) : (end-start);

    startOfTheBatch += start;
    endOfTheBatch += start;

    for(int i=startOfTheBatch ; i<endOfTheBatch ; ++i){
        biggestNumber[tid] = 0.0f;
        smalestNumber[tid] = FLT_MAX;

        for(int lp=tid ; lp<numberOfEntities ; lp+=blockDim.x){
            resultLine[idOfBatch][lp] = 0.0f;
        }
        __syncthreads();

        for(int d=0 ; d<dimensionOfEntity ; d+=256){
            if((tid < 256)&&(d+tid < dimensionOfEntity)){
                float* pElement = (float*)((char*)dataTable_device + (d+tid) * dataTable_Pitch) + i;
                entity[tid] = *pElement;
            }
            __syncthreads();

            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                float distance = 0.0;
                for(int k=d ; (k<dimensionOfEntity)&&(k<(d+256)) ; ++k){
                    float* pElement = (float*)((char*)dataTable_device + k * dataTable_Pitch) + lp;
                    float pElementVal = *pElement;
                    distance += abs(entity[k-d]-pElementVal);
                }
                resultLine[idOfBatch][lp] += distance;
            }
            __syncthreads();
        }

        for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
            float distance = resultLine[idOfBatch][lp];
            resultLineMarker[idOfBatch][lp] = 0;
            biggestNumber[tid] = max(biggestNumber[tid], ceil(distance));
            if(lp != i){
                smalestNumber[tid] = min(smalestNumber[tid], floor(distance));
            }
        }
        __syncthreads();
        if(tid == 0){
            resultLineMarker[idOfBatch][i] = OUT_OF_SOLUTION;
        }
        __syncthreads();

        if(tid < 32){
            for(int i=tid ; i<256 ; i+=32){
                biggestNumber[tid] = max(biggestNumber[tid], biggestNumber[i]);
                smalestNumber[tid] = min(smalestNumber[tid], smalestNumber[i]);
            }
        }
        if(tid == 0){
            #pragma unroll
            for(int c=0 ; c<32 ; ++c){
                biggestNumber[0] = max(biggestNumber[0], biggestNumber[c]);
                smalestNumber[0] = min(smalestNumber[0], smalestNumber[c]);
            }
        }
        __syncthreads();

        if(tid == 0){
            bias = smalestNumber[0];
            minValue = 0;
            maxValue = biggestNumber[0] - smalestNumber[0];
            maxValue = pow(2.0, ceil(log(maxValue+1.0)/log(2.0)));
            lengthOfBucket = (maxValue-minValue)/256.0;
            foundExactSolution = FALSE;
            limitOfLengthOfBucketExceeded = FALSE;
            alreadyFoundNumbers = 0;
            rewrittenNumbers = 0;
            complement = 0;
        }
        __syncthreads();

        while((foundExactSolution == FALSE) && (limitOfLengthOfBucketExceeded == FALSE)){
            hist[tid] = 0;
            interestingBucket = NON_OF_BUCKET_IN_INTEREST;
            __syncthreads();

            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                int marker = resultLineMarker[idOfBatch][lp];
                if(marker == 0){
                    int idOfBucketInHist = (resultLine[idOfBatch][lp]-bias-minValue)/lengthOfBucket;
                    atomicAdd(&hist[idOfBucketInHist], 1);
                    resultLineMarker[idOfBatch][lp] = idOfBucketInHist;
                }
            }
            __syncthreads();

            if(tid == 0){
                for(int k=1 ; k<256 ; ++k){
                    hist[k] += hist[k-1];
                }
            }
            __syncthreads();

            if((hist[tid]+alreadyFoundNumbers)>maximalNumbersToFind){
                atomicMin(&interestingBucket, tid);
            }

            if((tid == 0) && (alreadyFoundNumbers == maximalNumbersToFind) && (alreadyFoundNumbers == minimalNumbersToFind)){
                foundExactSolution = TRUE;
            }

            int tmpSum = hist[tid] + alreadyFoundNumbers;
            if((tmpSum>=minimalNumbersToFind) && (tmpSum<=maximalNumbersToFind)){
                foundExactSolution = TRUE;
            }

            if((tid == 0) && (lengthOfBucket < minimalSizeOfBucket)){
                limitOfLengthOfBucketExceeded = TRUE;
            }
            __syncthreads();

            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                int marker = resultLineMarker[idOfBatch][lp];
                if((marker < interestingBucket)&&(marker >= 0)){
                    resultLineMarker[idOfBatch][lp] = IN_SOLUTION;
                    atomicAdd(&alreadyFoundNumbers, 1);
                }else if((marker > interestingBucket)&&(marker < 256)){
                    resultLineMarker[idOfBatch][lp] = OUT_OF_SOLUTION;
                }else if(marker == interestingBucket){
                    resultLineMarker[idOfBatch][lp] = 0;
                }
            }

            if(tid == 0){
                bias = bias+interestingBucket*lengthOfBucket;
                minValue = 0.0;
                maxValue = lengthOfBucket;
                lengthOfBucket = (maxValue-minValue)/256.0;
            }
            __syncthreads();
        }
        __syncthreads();

        for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
            int marker = resultLineMarker[idOfBatch][lp];
            if(marker == IN_SOLUTION){
                int id = atomicAdd(&rewrittenNumbers, 1);
                neighboursDistance_device[i*numberOfNeighborsWithMistake+id] = resultLine[idOfBatch][lp];
                neighboursId_device[i*numberOfNeighborsWithMistake+id] = lp;
            }
        }
        __syncthreads();

        if((limitOfLengthOfBucketExceeded == TRUE)&&(foundExactSolution == FALSE)){
            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                int marker = resultLineMarker[idOfBatch][lp];
                if(marker == 0){
                    int id2 = atomicAdd(&complement, 1);
                    if((id2+alreadyFoundNumbers) < minimalNumbersToFind){
                        int id = atomicAdd(&rewrittenNumbers, 1);
                        neighboursDistance_device[i*numberOfNeighborsWithMistake+id] = resultLine[idOfBatch][lp];
                        neighboursId_device[i*numberOfNeighborsWithMistake+id] = lp;
                    }
                }
            }
        }
        __syncthreads();

        if(tid == 0){
            neighboursSize_device[i] = rewrittenNumbers;
        }
        __syncthreads();
    }
}

__global__
void kernelEuclidean(float* dataTable_device, size_t dataTable_Pitch, float* neighboursDistance_device, int* neighboursId_device,
                     float** resultLine, int** resultLineMarker, int* neighboursSize_device,
                     int dimensionOfEntity, int numberOfEntities, int numberOfNeighbors, int numberOfNeighborsWithMistake,
                     float mistakeShortage, float mistakeKindness, float minimalSizeOfBucket, int start, int end){
    __shared__ float biggestNumber[256];
    __shared__ float smalestNumber[256];
    __shared__ int hist[256];
    __shared__ float entity[256];
    __shared__ double minValue;
    __shared__ double maxValue;
    __shared__ double lengthOfBucket;
    __shared__ double bias;
    __shared__ int foundExactSolution;
    __shared__ int limitOfLengthOfBucketExceeded;
    __shared__ int alreadyFoundNumbers;
    __shared__ int interestingBucket;
    __shared__ int rewrittenNumbers;
    __shared__ int complement;

    int minimalNumbersToFind = float(numberOfNeighbors)*(1.0-mistakeShortage/100.0);
    int maximalNumbersToFind = float(numberOfNeighbors)*(1.0+mistakeKindness/100.0);

    int tid = threadIdx.x;
    int idOfBatch = blockIdx.x;
    int elementsPerBatch = ceil(float(end-start)/float(gridDim.x));
    int startOfTheBatch = elementsPerBatch*idOfBatch;
    int endOfTheBatch = elementsPerBatch*(idOfBatch+1) <= (end-start) ? elementsPerBatch*(idOfBatch+1) : (end-start);

    startOfTheBatch += start;
    endOfTheBatch += start;

    for(int i=startOfTheBatch ; i<endOfTheBatch ; ++i){
        biggestNumber[tid] = 0.0f;
        smalestNumber[tid] = FLT_MAX;

        for(int lp=tid ; lp<numberOfEntities ; lp+=blockDim.x){
            resultLine[idOfBatch][lp] = 0.0f;
        }
        __syncthreads();

        for(int d=0 ; d<dimensionOfEntity ; d+=256){
            if((tid < 256)&&(d+tid < dimensionOfEntity)){
                float* pElement = (float*)((char*)dataTable_device + (d+tid) * dataTable_Pitch) + i;
                entity[tid] = *pElement;
            }
            __syncthreads();
        
            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                float distance = 0.0;
                for(int k=d ; (k<dimensionOfEntity)&&(k<(d+256)) ; ++k){
                    float* pElement = (float*)((char*)dataTable_device + k * dataTable_Pitch) + lp;
                    float pElementVal = *pElement;
                    distance += (entity[k-d]-pElementVal)*(entity[k-d]-pElementVal);
                }
                resultLine[idOfBatch][lp] += distance;
            }
            __syncthreads();
        }

        for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
            float distance = sqrt(resultLine[idOfBatch][lp]);
            resultLine[idOfBatch][lp] = distance;
            resultLineMarker[idOfBatch][lp] = 0;
            biggestNumber[tid] = max(biggestNumber[tid], ceil(distance));
            if(lp != i){
                smalestNumber[tid] = min(smalestNumber[tid], floor(distance));
            }
        }
        __syncthreads();
        if(tid == 0){
            resultLineMarker[idOfBatch][i] = OUT_OF_SOLUTION;
        }
        __syncthreads();

        if(tid < 32){
            for(int i=tid ; i<256 ; i+=32){
                biggestNumber[tid] = max(biggestNumber[tid], biggestNumber[i]);
                smalestNumber[tid] = min(smalestNumber[tid], smalestNumber[i]);
            }
        }
        if(tid == 0){
            #pragma unroll
            for(int c=0 ; c<32 ; ++c){
                biggestNumber[0] = max(biggestNumber[0], biggestNumber[c]);
                smalestNumber[0] = min(smalestNumber[0], smalestNumber[c]);
            }
        }
        __syncthreads();
        if(tid == 0){
            bias = smalestNumber[0];
            minValue = 0;
            maxValue = biggestNumber[0] - smalestNumber[0];
            maxValue = pow(2.0, ceil(log(maxValue+1.0)/log(2.0)));
            lengthOfBucket = (maxValue-minValue)/256.0;
            foundExactSolution = FALSE;
            limitOfLengthOfBucketExceeded = FALSE;
            alreadyFoundNumbers = 0;
            rewrittenNumbers = 0;
            complement = 0;
        }
        __syncthreads();

        while((foundExactSolution == FALSE) && (limitOfLengthOfBucketExceeded == FALSE)){
            hist[tid] = 0;
            if(tid == 0){
                interestingBucket = NON_OF_BUCKET_IN_INTEREST;
            }
            __syncthreads();

            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                int marker = resultLineMarker[idOfBatch][lp];
                if(marker == 0){
                    int idOfBucketInHist = (resultLine[idOfBatch][lp]-bias-minValue)/lengthOfBucket;
                    atomicAdd(&hist[idOfBucketInHist], 1);
                    resultLineMarker[idOfBatch][lp] = idOfBucketInHist;
                }
            }
            __syncthreads();

            if(tid == 0){
                for(int k=1 ; k<256 ; ++k){
                    hist[k] += hist[k-1];
                }
            }
            __syncthreads();

            if((hist[tid]+alreadyFoundNumbers)>maximalNumbersToFind){
                atomicMin(&interestingBucket, tid);
            }

            if((tid == 0) && (alreadyFoundNumbers == maximalNumbersToFind) && (alreadyFoundNumbers == minimalNumbersToFind)){
                foundExactSolution = TRUE;
            }

            int tmpSum = hist[tid] + alreadyFoundNumbers;
            if((tmpSum>=minimalNumbersToFind) && (tmpSum<=maximalNumbersToFind)){
                foundExactSolution = TRUE;
            }

            if((tid == 0) && (lengthOfBucket < minimalSizeOfBucket)){
                limitOfLengthOfBucketExceeded = TRUE;
            }
            __syncthreads();

            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                int marker = resultLineMarker[idOfBatch][lp];
                if((marker < interestingBucket)&&(marker >= 0)){
                    resultLineMarker[idOfBatch][lp] = IN_SOLUTION;
                    atomicAdd(&alreadyFoundNumbers, 1);
                }else if((marker > interestingBucket)&&(marker < 256)){
                    resultLineMarker[idOfBatch][lp] = OUT_OF_SOLUTION;
                }else if(marker == interestingBucket){
                    resultLineMarker[idOfBatch][lp] = 0;
                }
            }
            __syncthreads();

            if(tid == 0){
                bias = bias+interestingBucket*lengthOfBucket;
                minValue = 0.0;
                maxValue = lengthOfBucket;
                lengthOfBucket = (maxValue-minValue)/256.0;
            }
            __syncthreads();
        }
        __syncthreads();

        for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
            int marker = resultLineMarker[idOfBatch][lp];
            if(marker == IN_SOLUTION){
                int id = atomicAdd(&rewrittenNumbers, 1);
                neighboursDistance_device[i*numberOfNeighborsWithMistake+id] = resultLine[idOfBatch][lp];
                neighboursId_device[i*numberOfNeighborsWithMistake+id] = lp;
            }
        }
        __syncthreads();

        if((limitOfLengthOfBucketExceeded == TRUE)&&(foundExactSolution == FALSE)){
            for(int lp = tid ; lp<numberOfEntities ; lp+=blockDim.x){
                int marker = resultLineMarker[idOfBatch][lp];
                if(marker == 0){
                    int id2 = atomicAdd(&complement, 1);
                    if((id2+alreadyFoundNumbers) < minimalNumbersToFind){
                        int id = atomicAdd(&rewrittenNumbers, 1);
                        neighboursDistance_device[i*numberOfNeighborsWithMistake+id] = resultLine[idOfBatch][lp];
                        neighboursId_device[i*numberOfNeighborsWithMistake+id] = lp;
                    }
                }
            }
        }
        __syncthreads(); 

        if(tid == 0){
            neighboursSize_device[i] = rewrittenNumbers;
        }
        __syncthreads();
    }
}

std::string trim(std::string const& str){
    if(str.empty())
        return str;

    std::size_t firstScan = str.find_first_not_of(' ');
    std::size_t first     = firstScan == std::string::npos ? str.length() : firstScan;
    std::size_t last      = str.find_last_not_of(' ');
    return str.substr(first, last-first+1);
}

EuclideanDistanceMatrixGPU::EuclideanDistanceMatrixGPU(){
    this->numberOfBlocksPerMultiprocessors = 30;
    this->subTaskSize = 10000;
    typeOfDistance = DISTANCE_EUCLIDEAN;
    this->debugMode = false;
}

EuclideanDistanceMatrixGPU::EuclideanDistanceMatrixGPU(bool debugMode){
    this->numberOfBlocksPerMultiprocessors = 30;
    this->subTaskSize = 10000;
    typeOfDistance = DISTANCE_EUCLIDEAN;
    this->debugMode = debugMode;
}

EuclideanDistanceMatrixGPU::~EuclideanDistanceMatrixGPU(){

}

void EuclideanDistanceMatrixGPU::setDataFile(std::string nameOfFile){
    this->inputFile = nameOfFile;
}

bool EuclideanDistanceMatrixGPU::loadData(){
    std::ifstream myfile;
    myfile.open(this->inputFile.c_str());
    if (myfile.is_open()){
        std::cout<<"The file with data is opened.\n";
    }else{
        std::cout<<"The file with data can not be opened.\n";
        return true;
    }
    std::string line;

    std::getline(myfile, line);
    std::getline(myfile, line);

    int idOfEntity = 0;
    char* lineChar;
    while ((std::getline(myfile, line))&&(idOfEntity<numberOfEntities)){
        std::vector<std::string> cuttedString;
        lineChar = new char[line.length() + 1];
        std::strcpy(lineChar, line.c_str());

        std::string str;
        char* pch = strtok(lineChar,",");
        while (pch != NULL){
            str = std::string(pch);
            str = trim(str);
            cuttedString.push_back(str);
            pch = strtok (NULL, ",");
        }

        delete [] lineChar;

        if(klaster){
            for(int i=0 ; i<cuttedString.size()-1 ; ++i){
                this->dataTable_host[idOfEntity+numberOfEntities*i] = atof(cuttedString[i].c_str());
            }
            this->dataTableId_host[idOfEntity] = atoi(cuttedString[cuttedString.size()-1].c_str());
        }else{
            for(int i=0 ; i<cuttedString.size() ; ++i){
                this->dataTable_host[idOfEntity+numberOfEntities*i] = atof(cuttedString[i].c_str());
            }
        }
        idOfEntity++;
    }
    return false;
}

bool EuclideanDistanceMatrixGPU::initialize(int numberOfEntities, int dimensionOfEntity, int numberOfNeighbors,
                                            float mistakeShortage, float mistakeKindness, float minimalSizeOfBucket,
                                            std::vector<int> devices, int typeOfDistance, bool klaster){
    this->typeOfDistance = typeOfDistance;
    this->klaster = klaster;
    this->numberOfEntities = numberOfEntities;
    this->numberOfNeighbors = numberOfNeighbors;
    this->dimensionOfEntity = dimensionOfEntity;

    this->mistakeKindness = mistakeKindness;
    this->mistakeShortage = mistakeShortage;
    this->minimalSizeOfBucket = minimalSizeOfBucket;

    this->numberOfNeighborsWithMistake = numberOfNeighbors*(1.0+mistakeKindness/100.0);

    this->devices = devices;

    bool error = false;

    error |= cuCall(cudaSetDevice(devices[0]));
    cudaDeviceProp devProp;
    error |= cuCall(cudaGetDeviceProperties(&devProp, devices[0]));
    this->numberOfMultiprocessors = devProp.multiProcessorCount;

    error |= cuCall(cudaSetDevice(devices[0]));
    error |= cuCall(cudaHostAlloc((void**)&dataTable_host, numberOfEntities*dimensionOfEntity*sizeof(float), cudaHostAllocPortable));
    error |= cuCall(cudaHostAlloc((void**)&dataTableId_host, numberOfEntities*sizeof(int), cudaHostAllocPortable));

    for(int i=0 ; i<devices.size() ; ++i){
        error |= cuCall(cudaSetDevice(devices[i]));
        error |= cuCall(cudaMallocPitch((void**)&dataTable_device[i], &dataTable_Pitch[i], numberOfEntities*sizeof(float), dimensionOfEntity));

        error |= cuCall(cudaMallocHost((void**)&neighboursDistance_host[i], numberOfNeighborsWithMistake*numberOfEntities*sizeof(float)));
        error |= cuCall(cudaMalloc((void**)&neighboursDistance_device[i], numberOfNeighborsWithMistake*numberOfEntities*sizeof(float)));

        error |= cuCall(cudaMallocHost((void**)&neighboursId_host[i], numberOfNeighborsWithMistake*numberOfEntities*sizeof(int)));
        error |= cuCall(cudaMalloc((void**)&neighboursId_device[i], numberOfNeighborsWithMistake*numberOfEntities*sizeof(int)));

        error |= cuCall(cudaMallocHost((void**)&neighboursSize_host[i], numberOfEntities*sizeof(int)));
        error |= cuCall(cudaMalloc((void**)&neighboursSize_device[i], numberOfEntities*sizeof(int)));

        resultLine_tmp[i] = new float*[numberOfBlocksPerMultiprocessors*numberOfMultiprocessors];
        for(int ii=0 ; ii<numberOfBlocksPerMultiprocessors*numberOfMultiprocessors ; ++ii){
            error |= cuCall(cudaMalloc((void**)&resultLine_tmp[i][ii], numberOfEntities*sizeof(float)));
        }
        error |= cuCall(cudaMalloc((void**)&resultLine[i], numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*sizeof(float*)));
        error |= cuCall(cudaMemcpy((void*)resultLine[i], (void*)resultLine_tmp[i], numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*sizeof(float*), cudaMemcpyHostToDevice));

        resultLine2_tmp[i] = new float*[numberOfBlocksPerMultiprocessors*numberOfMultiprocessors];
        for(int ii=0 ; ii<numberOfBlocksPerMultiprocessors*numberOfMultiprocessors ; ++ii){
            error |= cuCall(cudaMalloc((void**)&resultLine2_tmp[i][ii], numberOfEntities*sizeof(float)));
        }
        error |= cuCall(cudaMalloc((void**)&resultLine2[i], numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*sizeof(float*)));
        error |= cuCall(cudaMemcpy((void*)resultLine2[i], (void*)resultLine2_tmp[i], numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*sizeof(float*), cudaMemcpyHostToDevice));

        resultLine3_tmp[i] = new float*[numberOfBlocksPerMultiprocessors*numberOfMultiprocessors];
        for(int ii=0 ; ii<numberOfBlocksPerMultiprocessors*numberOfMultiprocessors ; ++ii){
            error |= cuCall(cudaMalloc((void**)&resultLine3_tmp[i][ii], numberOfEntities*sizeof(float)));
        }
        error |= cuCall(cudaMalloc((void**)&resultLine3[i], numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*sizeof(float*)));
        error |= cuCall(cudaMemcpy((void*)resultLine3[i], (void*)resultLine3_tmp[i], numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*sizeof(float*), cudaMemcpyHostToDevice));

        resultLineMarker_tmp[i] = new int*[numberOfBlocksPerMultiprocessors*numberOfMultiprocessors];
        for(int ii=0 ; ii<numberOfBlocksPerMultiprocessors*numberOfMultiprocessors ; ++ii){
            error |= cuCall(cudaMalloc((void**)&resultLineMarker_tmp[i][ii], numberOfEntities*sizeof(int)));
        }
        error |= cuCall(cudaMalloc((void**)&resultLineMarker[i], numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*sizeof(int*)));
        error |= cuCall(cudaMemcpy((void*)resultLineMarker[i], (void*)resultLineMarker_tmp[i], numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*sizeof(int*), cudaMemcpyHostToDevice));

        error |= cuCall(cudaStreamCreate(&executionStreams[i]));
        error |= cuCall(cudaEventCreate(&startEvents[i]));
        error |= cuCall(cudaEventCreate(&stopEvents[i]));
    }
    error |= cuCall(cudaSetDevice(devices[0]));
    error |= loadData();

    for(int i=0 ; i<devices.size() ; ++i){
        error |= cuCall(cudaSetDevice(devices[i]));
        error |= cuCall(cudaMemcpy2D((void*)dataTable_device[i], dataTable_Pitch[i], (void*)dataTable_host, numberOfEntities*sizeof(float),
                                     numberOfEntities*sizeof(float), dimensionOfEntity, cudaMemcpyHostToDevice));
    }
    error |= cuCall(cudaSetDevice(devices[0]));

    int elementsPerBatch = ceil(float(numberOfEntities)/float(devices.size()));
    for(int i=0 ; i<devices.size() ; ++i){
        int startOfTheBatch = elementsPerBatch*i;
        int endOfTheBatch = elementsPerBatch*(i+1) <= numberOfEntities ? elementsPerBatch*(i+1) : numberOfEntities;
        Partition p = {startOfTheBatch, endOfTheBatch};
        partition[i] = p;
    }

    for(int i=0 ; i<devices.size() ; ++i){
        Partition p = partition[i];
        int pSize = p.end-p.start;
        std::vector<SubTask> vesSubTasks;
        int subTasksNumber = floor(float(pSize)/float(subTaskSize));
        int bias = p.start;
        for(int k=0 ; k<subTasksNumber ; ++k){
            SubTask subTask = {bias, bias+subTaskSize, false};
            vesSubTasks.push_back(subTask);
            bias += subTaskSize;
        }
        if(pSize % subTaskSize != 0){
            SubTask subTask = {bias, p.end, false};
            vesSubTasks.push_back(subTask);
        }
        taskPartitioning[i] = vesSubTasks;
    }

    return error;
}

bool EuclideanDistanceMatrixGPU::deinitialize(){
    bool error = false;

    for(int i=0 ; i<devices.size() ; ++i){
        error |= cuCall(cudaSetDevice(devices[i]));
        error |= cuCall(cudaDeviceSynchronize());
    }

    error |= cuCall(cudaSetDevice(devices[0]));
    error |= cuCall(cudaFreeHost((void*)dataTable_host));
    error |= cuCall(cudaFreeHost((void*)dataTableId_host));

    for(int i=0 ; i<devices.size() ; ++i){
        error |= cuCall(cudaSetDevice(devices[i]));

        error |= cuCall(cudaFree((void*)dataTable_device[i]));

        error |= cuCall(cudaFreeHost((void*)neighboursDistance_host[i]));
        error |= cuCall(cudaFree((void*)neighboursDistance_device[i]));

        error |= cuCall(cudaFreeHost((void*)neighboursId_host[i]));
        error |= cuCall(cudaFree((void*)neighboursId_device[i]));

        error |= cuCall(cudaFreeHost((void*)neighboursSize_host[i]));
        error |= cuCall(cudaFree((void*)neighboursSize_device[i]));

        for(int ii=0 ; ii<numberOfBlocksPerMultiprocessors*numberOfMultiprocessors ; ++ii){
            error |= cuCall(cudaFree((void*)resultLine_tmp[i][ii]));
        }
        delete [] resultLine_tmp[i];
        error |= cuCall(cudaFree((void*)resultLine[i]));

        for(int ii=0 ; ii<numberOfBlocksPerMultiprocessors*numberOfMultiprocessors ; ++ii){
            error |= cuCall(cudaFree((void*)resultLine2_tmp[i][ii]));
        }
        delete [] resultLine2_tmp[i];
        error |= cuCall(cudaFree((void*)resultLine2[i]));

        for(int ii=0 ; ii<numberOfBlocksPerMultiprocessors*numberOfMultiprocessors ; ++ii){
            error |= cuCall(cudaFree((void*)resultLine3_tmp[i][ii]));
        }
        delete [] resultLine3_tmp[i];
        error |= cuCall(cudaFree((void*)resultLine3[i]));

        for(int ii=0 ; ii<numberOfBlocksPerMultiprocessors*numberOfMultiprocessors ; ++ii){
            error |= cuCall(cudaFree((void*)resultLineMarker_tmp[i][ii]));
        }
        delete [] resultLineMarker_tmp[i];
        error |= cuCall(cudaFree((void*)resultLineMarker[i]));

        error |= cuCall(cudaStreamDestroy(executionStreams[i]));
        error |= cuCall(cudaEventDestroy(startEvents[i]));
        error |= cuCall(cudaEventDestroy(stopEvents[i]));

        error |= cuCall(cudaDeviceReset());
    }
    error |= cuCall(cudaSetDevice(devices[0]));

    return error;
}

bool EuclideanDistanceMatrixGPU::calculate(){
    dim3 grid(this->numberOfBlocksPerMultiprocessors*this->numberOfMultiprocessors, 1);
    dim3 block(256, 1);

    bool error = false;

    if(typeOfDistance == DISTANCE_EUCLIDEAN){
        for(int i=0 ; i<devices.size() ; ++i){
            error |= cuCall(cudaSetDevice(devices[i]));
            error |= cuCall(cudaEventRecord(startEvents[i], executionStreams[i]));
        }

        for(int i=0 ; i<devices.size() ; ++i){
            std::cout<<"The device "<<devices[i]<<" is calculating the neighbours for: "<<partition[i].start<<" - "<<partition[i].end-1<<"\n";
        }

        bool jobDone = false;
        while(jobDone == false){
            jobDone = true;
            for(int i=0 ; i <devices.size() ; ++i){
                if(taskPartitioning[i].size() > 0){
					error |= cuCall(cudaSetDevice(devices[i]));
                    SubTask subTask = taskPartitioning[i][0];
                    std::cout<<"The device "<<devices[i]<<" is calculating for points: "<<subTask.startPoint<<" - "<<subTask.endPoint-1<<"\n";
                    kernelEuclidean<<<grid, block, 0, executionStreams[i]>>>(dataTable_device[i], dataTable_Pitch[i], neighboursDistance_device[i],
                                                                             neighboursId_device[i], resultLine[i], resultLineMarker[i], neighboursSize_device[i],
                                                                             dimensionOfEntity, numberOfEntities, numberOfNeighbors, numberOfNeighborsWithMistake,
                                                                             mistakeShortage, mistakeKindness, minimalSizeOfBucket, subTask.startPoint, subTask.endPoint);
                }
            }
            for(int i=0 ; i <devices.size() ; ++i){
                if(taskPartitioning[i].size() > 0){
                    taskPartitioning[i].erase(taskPartitioning[i].begin());
                    cudaStreamSynchronize(executionStreams[i]);
                }
            }
            for(int i=0 ; i <devices.size() ; ++i){
                if(taskPartitioning[i].size() > 0){
                    jobDone = false;
                }
            }
        }

        for(int i=0 ; i<devices.size() ; ++i){
            error |= cuCall(cudaSetDevice(devices[i]));
            error |= cuCall(cudaEventRecord(stopEvents[i], executionStreams[i]));
        }

        for(int i=0 ; i<devices.size() ; ++i){
            error |= cuCall(cudaSetDevice(devices[i]));
            error |= cuCall(cudaEventSynchronize(stopEvents[i]));
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]);
            std::cout<<"The device "<<devices[i]<<": has done task in: "<<milliseconds<<" ms\n";
        }
    }else if(typeOfDistance == DISTANCE_TAXICAB){
        for(int i=0 ; i<devices.size() ; ++i){
            error |= cuCall(cudaSetDevice(devices[i]));
            error |= cuCall(cudaEventRecord(startEvents[i], executionStreams[i]));
        }

        for(int i=0 ; i<devices.size() ; ++i){
            std::cout<<"The device "<<devices[i]<<" is calculating the neighbours for: "<<partition[i].start<<" - "<<partition[i].end-1<<"\n";
        }

        //wywolanie wyliczania
        bool jobDone = false;
        while(jobDone == false){
            jobDone = true;
            for(int i=0 ; i <devices.size() ; ++i){
                if(taskPartitioning[i].size() > 0){
					error |= cuCall(cudaSetDevice(devices[i]));
                    SubTask subTask = taskPartitioning[i][0];
                    std::cout<<"The device "<<devices[i]<<" is calculating for points: "<<subTask.startPoint<<" - "<<subTask.endPoint-1<<"\n";
                    kernelTaxicab<<<grid, block, 0, executionStreams[i]>>>(dataTable_device[i], dataTable_Pitch[i], neighboursDistance_device[i],
                                                                           neighboursId_device[i], resultLine[i], resultLineMarker[i], neighboursSize_device[i],
                                                                           dimensionOfEntity, numberOfEntities, numberOfNeighbors, numberOfNeighborsWithMistake,
                                                                           mistakeShortage, mistakeKindness, minimalSizeOfBucket, subTask.startPoint, subTask.endPoint);
                }
            }
            for(int i=0 ; i <devices.size() ; ++i){
                if(taskPartitioning[i].size() > 0){
                    taskPartitioning[i].erase(taskPartitioning[i].begin());
                    cudaStreamSynchronize(executionStreams[i]);
                }
            }
            for(int i=0 ; i <devices.size() ; ++i){
                if(taskPartitioning[i].size() > 0){
                    jobDone = false;
                }
            }
        }

        for(int i=0 ; i<devices.size() ; ++i){
            error |= cuCall(cudaSetDevice(devices[i]));
            error |= cuCall(cudaEventRecord(stopEvents[i], executionStreams[i]));
        }

        for(int i=0 ; i<devices.size() ; ++i){
            error |= cuCall(cudaSetDevice(devices[i]));
            error |= cuCall(cudaEventSynchronize(stopEvents[i]));
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]);
            std::cout<<"The device "<<devices[i]<<": has done task in: "<<milliseconds<<" ms\n";
        }
    }else if(typeOfDistance == DISTANCE_COSINE){
        for(int i=0 ; i<devices.size() ; ++i){
            error |= cuCall(cudaSetDevice(devices[i]));
            error |= cuCall(cudaEventRecord(startEvents[i], executionStreams[i]));
        }

        for(int i=0 ; i<devices.size() ; ++i){
            std::cout<<"The device "<<devices[i]<<" is calculating the neighbours for: "<<partition[i].start<<" - "<<partition[i].end-1<<"\n";
        }

        //wywolanie wyliczania
        bool jobDone = false;
        while(jobDone == false){
            jobDone = true;
            for(int i=0 ; i <devices.size() ; ++i){
                if(taskPartitioning[i].size() > 0){
					error |= cuCall(cudaSetDevice(devices[i]));
                    SubTask subTask = taskPartitioning[i][0];
                    std::cout<<"The device "<<devices[i]<<" is calculating for points: "<<subTask.startPoint<<" - "<<subTask.endPoint-1<<"\n";
                    kernelCosine<<<grid, block, 0, executionStreams[i]>>>(dataTable_device[i], dataTable_Pitch[i], neighboursDistance_device[i],
                                                                          neighboursId_device[i], resultLine[i], resultLine2[i], resultLine3[i],
                                                                          resultLineMarker[i], neighboursSize_device[i], dimensionOfEntity, numberOfEntities,
                                                                          numberOfNeighbors, numberOfNeighborsWithMistake, mistakeShortage, mistakeKindness, minimalSizeOfBucket,
                                                                          subTask.startPoint, subTask.endPoint);
                }
            }
            for(int i=0 ; i <devices.size() ; ++i){
                if(taskPartitioning[i].size() > 0){
                    taskPartitioning[i].erase(taskPartitioning[i].begin());
                    cudaStreamSynchronize(executionStreams[i]);
                }
            }
            for(int i=0 ; i <devices.size() ; ++i){
                if(taskPartitioning[i].size() > 0){
                    jobDone = false;
                }
            }
        }

        for(int i=0 ; i<devices.size() ; ++i){
            error |= cuCall(cudaSetDevice(devices[i]));
            error |= cuCall(cudaEventRecord(stopEvents[i], executionStreams[i]));
        }

        for(int i=0 ; i<devices.size() ; ++i){
            error |= cuCall(cudaSetDevice(devices[i]));
            error |= cuCall(cudaEventSynchronize(stopEvents[i]));
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]);
            std::cout<<"The device "<<devices[i]<<": has done task in: "<<milliseconds<<" ms\n";
        }
    }else{
        std::cout<<"We do not have such type of distance\n";
    }

    return error;
}

void EuclideanDistanceMatrixGPU::setResultsFile(std::string nameOfFile){
    this->outputFile = nameOfFile;
}

template <typename T> std::string tostr(const T& t) {
   std::ostringstream os;
   os<<t;
   return os.str();
}

bool EuclideanDistanceMatrixGPU::saveResultToResultFile(){
    bool error = false;
    for(int i=0 ; i<devices.size() ; ++i){
        error |= cuCall(cudaSetDevice(devices[i]));
        error |= cuCall(cudaMemcpyAsync((void*)neighboursDistance_host[i], (void*)neighboursDistance_device[i],
                                        numberOfNeighborsWithMistake*numberOfEntities*sizeof(float), cudaMemcpyDeviceToHost, executionStreams[i]));
        error |= cuCall(cudaMemcpyAsync((void*)neighboursId_host[i], (void*)neighboursId_device[i],
                                        numberOfNeighborsWithMistake*numberOfEntities*sizeof(int), cudaMemcpyDeviceToHost, executionStreams[i]));
        error |= cuCall(cudaMemcpyAsync((void*)neighboursSize_host[i], (void*)neighboursSize_device[i],
                                        numberOfEntities*sizeof(int), cudaMemcpyDeviceToHost, executionStreams[i]));
    }
    for(int i=0 ; i<devices.size() ; ++i){
        error |= cuCall(cudaSetDevice(devices[i]));
        error |= cuCall(cudaStreamSynchronize(executionStreams[i]));
    }

    std::ofstream ofs;
    ofs.open(outputFile.c_str(), std::ofstream::trunc | std::ofstream::binary);
    std::ofstream ofsDebug;
    if(debugMode){
        ofsDebug.open((outputFile+"DEBUG").c_str(), std::ofstream::trunc | std::ofstream::binary);
    }
    bool validationSuccess = true;
    std::ofstream ofsValidation;
    ofsValidation.open((outputFile+"VALIDATION").c_str(), std::ofstream::trunc | std::ofstream::binary);
    if(ofs.is_open()){
        ofs<<numberOfEntities<<";"<< numberOfNeighbors<<";"<<sizeof(long)<<"\n";
        long l = 0x01020304;
        ofs.write((char*)&l, sizeof(long));
        //zapisywanie punktow
        for(int i=0 ; i<devices.size() ; ++i){
            for(int lp=partition[i].start ; lp<partition[i].end ; ++lp){
                std::vector<DataPoint> liczbyNear;
                for(int c=0 ; c<neighboursSize_host[i][lp] ; ++c){
                    DataPoint dp = {neighboursId_host[i][lp*numberOfNeighborsWithMistake+c], neighboursDistance_host[i][lp*numberOfNeighborsWithMistake+c]};
                    liczbyNear.push_back(dp);
                }
                if(neighboursSize_host[i][lp] != numberOfNeighborsWithMistake){
                    if(ofsValidation.is_open()){
                        ofsValidation<<"Something wrong for NEAR: For "<<lp<<" : we have "<<neighboursSize_host[i][lp]<<" but we should have "<<numberOfNeighborsWithMistake<<"\n";
                        validationSuccess = false;
                    }
                }
				if(typeOfDistance == DISTANCE_COSINE){
					std::sort(liczbyNear.begin(), liczbyNear.end(), compareByLengthMinCosine);
				}else{
					std::sort(liczbyNear.begin(), liczbyNear.end(), compareByLengthMin);
				}
                for(std::vector<DataPoint>::iterator it = liczbyNear.begin() ; it != liczbyNear.end() ; ++it){
                    DataPoint f = *it;
                    ofs.write((char*)&f.id, sizeof(long));
                    if((debugMode)&&(ofsDebug.is_open())){
                        ofsDebug<<"NEAR: <"<<lp<<", "<<f.id<<">("<<f.distance<<") ";
                    }
                }
                if((debugMode)&&(ofsDebug.is_open())){
                    ofsDebug<<";\n";
                }
            }
        }
        ofs.close();
        if((debugMode)&&(ofsDebug.is_open())){
            ofsDebug.close();
        }
        if(ofsValidation.is_open()){
            if(validationSuccess){
                ofsValidation<<"Everything is OK.";
            }
            ofsValidation.close();
        }
    }else{
        std::cout <<"Can not open the file for saving result.\n";
        error |= true;
    }
    return error;
}
