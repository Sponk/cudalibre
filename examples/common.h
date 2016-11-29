#pragma once

/**
 * @brief Works like an assertion and quits the program on error.
 */
#define CUDA_CHECK(x) { cudaError_t err = (x); \
						if(err != cudaSuccess) \
						{ \
								std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
								exit(1);\
						}\
					}

#define CUDA_CHECK_LAST CUDA_CHECK(cudaGetLastError())