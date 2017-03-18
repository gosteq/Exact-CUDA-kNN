#ifndef EUCLIDEANDISTANCEMATRIXGPU_H
#define EUCLIDEANDISTANCEMATRIXGPU_H

#include <iostream>
#include "cucall.h"
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>

#define DISTANCE_EUCLIDEAN 0
#define DISTANCE_TAXICAB 1
#define DISTANCE_COSINE 2

typedef struct __align__(8){
    int start;
    int end;
} Partition;

struct SubTask{
    int startPoint;
    int endPoint;
    bool doneJob;
};

class EuclideanDistanceMatrixGPU{
    public:
        EuclideanDistanceMatrixGPU();
        EuclideanDistanceMatrixGPU(bool debugMode);
        ~EuclideanDistanceMatrixGPU();

        void setDataFile(std::string nameOfFile);
        bool initialize(int numberOfEntities, int dimensionOfEntity, int numberOfNeighbors,
                        float mistakeShortage, float mistakeKindness, float minimalSizeOfBucket,
                        std::vector<int> devices, int typeOfDistance, bool klaster);
        bool deinitialize();
        bool calculate();
        void setResultsFile(std::string nameOfFile);
        bool saveResultToResultFile();

    private:
        bool loadData();

        int typeOfDistance;
        bool klaster;
        bool debugMode;

        std::string inputFile;
        std::string outputFile;

        float mistakeShortage;
        float mistakeKindness;
        float minimalSizeOfBucket;

        int dimensionOfEntity;
        int numberOfEntities;
        int numberOfNeighbors;
        int numberOfNeighborsWithMistake;

        float* dataTable_host;
        int* dataTableId_host;
        std::map<int, float*> dataTable_device;
        std::map<int, size_t> dataTable_Pitch;

        std::map<int, float*> neighboursDistance_host;
        std::map<int, float*> neighboursDistance_device;

        std::map<int, int*> neighboursId_host;
        std::map<int, int*> neighboursId_device;

        std::map<int, int*> neighboursSize_host;
        std::map<int, int*> neighboursSize_device;

        std::map<int, float**> resultLine;
        std::map<int, float**> resultLine_tmp;
        std::map<int, float**> resultLine2;
        std::map<int, float**> resultLine2_tmp;
        std::map<int, float**> resultLine3;
        std::map<int, float**> resultLine3_tmp;
        std::map<int, int**> resultLineMarker;
        std::map<int, int**> resultLineMarker_tmp;

        std::map<int, cudaStream_t> executionStreams;
        std::map<int, cudaEvent_t> startEvents;
        std::map<int, cudaEvent_t> stopEvents;

        std::map<int, Partition> partition;

        int numberOfBlocksPerMultiprocessors;
        int numberOfMultiprocessors;
        std::vector<int> devices;

        int subTaskSize;
        std::map<int, std::vector<SubTask> > taskPartitioning;
};

#endif // EUCLIDEANDISTANCEMATRIXGPU_H
