#ifndef CUCALL_H_
#define CUCALL_H_

#include <iostream>  // NOLINT(readability/streams)

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

inline bool cudaCheck(cudaError_t ret, const char *fileName, unsigned int lineNo) {
    if (ret != cudaSuccess) {
        std::cout << "CUDA error in " << fileName << ":" << lineNo << std::endl
                  << "\t (" << cudaGetErrorString(ret) << ")" << std::endl;
    }

    return ret != cudaSuccess;
}

inline bool curandCheck(curandStatus_t ret, const char *fileName, unsigned int lineNo) {
    if (ret != CURAND_STATUS_SUCCESS) {
        std::cout << "CURAND error in " << fileName << ":" << lineNo
                  << " (return code: " << ret << ")" << std::endl;
    }

    return ret != CURAND_STATUS_SUCCESS;
}

inline bool cusolverCheck(cusolverStatus_t ret, const char *fileName, unsigned int lineNo) {
    if (ret != CUSOLVER_STATUS_SUCCESS) {
        std::cout << "CUSOLVER error in " << fileName << ":" << lineNo
                  << " (return code: " << ret << ")" << std::endl;
    }

    return ret != CUSOLVER_STATUS_SUCCESS;
}

inline bool cusparseCheck(cusparseStatus_t ret, const char *fileName, unsigned int lineNo) {
    if (ret != CUSPARSE_STATUS_SUCCESS) {
        std::cout << "CUSPARSE error in " << fileName << ":" << lineNo
                  << " (return code: " << ret << ")" << std::endl;
    }

    return ret != CUSPARSE_STATUS_SUCCESS;
}


#define cuCall(err)       cudaCheck(err, __FILE__, __LINE__)
#define curandCall(err)   curandCheck(err, __FILE__, __LINE__)
#define cusolverCall(err)   cusolverCheck(err, __FILE__, __LINE__)
#define cusparseCall(err)   cusparseCheck(err, __FILE__, __LINE__)

#endif  // CUCALL_H_
