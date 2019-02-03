
#include "..\Core.h"

// CUDA HEADERS
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "..\Utility\LogCore.h"

/*
 * Extracted from "helper_cuda.h from NVIDIA Corporation CUDA Examples,
 * to use the Debug and custom debug functions"
**/
namespace CUDA {

	template <typename T>
	void CheckFunction(T Result, WChar const *const FunctionName, const char *const FileName, int const Line) {
		if (Result) {
			Debug::Log(
				Debug::LogCritical, L"CUDA error at %s:%d code=%d(%s) \"%s\" \n", CharToWChar(FileName), Line,
				static_cast<unsigned int>(Result), cudaGetErrorName(Result), FunctionName
			);

			cudaDeviceReset();
			// Make sure we call CUDA Device Reset before exiting
			exit(EXIT_FAILURE);
		}
	}

	// This will output the proper CUDA error strings in the event
	// that a CUDA host call returns an error
#define CheckCudaErrors(func) CheckFunction((func), L#func, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define GetLastCudaError(msg) __GetLastCudaError(msg, __FILE__, __LINE__)

	inline void __GetLastCudaError(const char *ErrorMessage, const char *FileName, const int Line) {
		cudaError_t CUDAError = cudaGetLastError();

		if (cudaSuccess != CUDAError) {
			Debug::Log(
				Debug::LogCritical,
				L"%s(%i) : CUDA error : %s : (%d) %s.\n",
				CharToWChar(FileName), Line, CharToWChar(ErrorMessage), static_cast<int>(CUDAError),
				CharToWChar(cudaGetErrorString(CUDAError))
			);

			cudaDeviceReset();
			// Make sure we call CUDA Device Reset before exiting
			exit(EXIT_FAILURE);
		}
	}

	// Beginning of GPU Architecture definitions
	inline int _ConvertSMVer2Cores(int major, int minor) {
		// Defines for GPU Architecture types (using the SM version to determine
		// the # of cores per SM
		typedef struct {
			int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
			// and m = SM minor version
			int Cores;
		} sSMtoCores;

		sSMtoCores nGpuArchCoresPerSM[] = {
			{0x30, 192},
			{0x32, 192},
			{0x35, 192},
			{0x37, 192},
			{0x50, 128},
			{0x52, 128},
			{0x53, 128},
			{0x60,  64},
			{0x61, 128},
			{0x62, 128},
			{0x70,  64},
			{0x72,  64},
			{0x75,  64},
			{-1, -1} };

		int index = 0;

		while (nGpuArchCoresPerSM[index].SM != -1) {
			if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
				return nGpuArchCoresPerSM[index].Cores;
			}

			index++;
		}

		// If we don't find the values, we default use the previous one
		// to run properly
		Debug::Log(
			Debug::LogError,
			L"MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n",
			major, minor, nGpuArchCoresPerSM[index - 1].Cores
		);
		return nGpuArchCoresPerSM[index - 1].Cores;
	}

	// Returns the best GPU (maximum GFLOPS)
	inline int GetMaxGflopsDeviceId() {
		int current_device = 0, sm_per_multiproc = 0;
		int max_perf_device = 0;
		int device_count = 0;
		int devices_prohibited = 0;

		uint64_t max_compute_perf = 0;
		cudaDeviceProp deviceProp;
		CheckCudaErrors(cudaGetDeviceCount(&device_count));

		if (device_count == 0) {
			Debug::Log(
				Debug::LogCritical,
				L"GetMaxGflopsDeviceId(): No devices supporting CUDA.\n"
			);
			exit(EXIT_FAILURE);
		}

		// Find the best CUDA capable GPU device
		current_device = 0;

		while (current_device < device_count) {
			cudaGetDeviceProperties(&deviceProp, current_device);

			// If this GPU is not running on Compute Mode prohibited,
			// then we can add it to the list
			if (deviceProp.computeMode != cudaComputeModeProhibited) {
				if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
					sm_per_multiproc = 1;
				} else {
					sm_per_multiproc =
						_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
				}

				uint64_t compute_perf = (uint64_t)deviceProp.multiProcessorCount *
					sm_per_multiproc * deviceProp.clockRate;

				if (compute_perf > max_compute_perf) {
					max_compute_perf = compute_perf;
					max_perf_device = current_device;
				}
			} else {
				devices_prohibited++;
			}

			++current_device;
		}

		if (devices_prohibited == device_count) {
			Debug::Log(
				Debug::LogCritical,
				L"GetMaxGflopsDeviceId(): All devices have compute mode prohibited.\n"
			);
			exit(EXIT_FAILURE);
		}

		return max_perf_device;
	}

	inline int FindCudaDevice(int Argc, const char **Argv) {
		cudaDeviceProp DeviceProperties;
		int DeviceID = 0;

		// Otherwise pick the device with highest Gflops/s
		DeviceID = GetMaxGflopsDeviceId();
		CheckCudaErrors(cudaSetDevice(DeviceID));
		CheckCudaErrors(cudaGetDeviceProperties(&DeviceProperties, DeviceID));
		Debug::Log(Debug::LogNormal, L"GPU Device %d: \"%s\" with compute capability %d.%d\n\n", DeviceID,
			CharToWChar(DeviceProperties.name), DeviceProperties.major, DeviceProperties.minor);

		return DeviceID;
	}
}