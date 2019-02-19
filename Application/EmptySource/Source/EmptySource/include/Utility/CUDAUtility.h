
#include "..\Core.h"
#include "..\Utility\LogCore.h"

// CUDA HEADERS
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_functions.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>

/*
 * Extracted from "helper_cuda.h from NVIDIA Corporation CUDA Samples,
 * and modified to use the Debug and custom debug functions"
**/
namespace CUDA {

	template <typename T>
	void __CheckFunction(T Result, WChar const *const FunctionName, const char *const FileName, int const Line) {
		if (Result) {
			Debug::Log(
				Debug::LogCritical, L"CUDA error at %s:%d code=%d(%s) '%s'", CharToWChar(FileName), Line,
				static_cast<unsigned int>(Result), CharToWChar(cudaGetErrorName(Result)), FunctionName
			);

			// --- Make sure we call CUDA Device Reset before exiting
			cudaDeviceReset();
			Debug::Log(Debug::LogNormal, L"Press any key to close...");
			_getch();
			exit(EXIT_FAILURE);
		}
	}

	/* This will output the proper CUDA error strings in the event
	 *  that a CUDA host call returns an error
	 */
#define Check(func) __CheckFunction((func), L#func, __FILE__, __LINE__)

	//* This will output the proper error string when calling cudaGetLastError
#define GetLastCudaError(msg) __GetLastCudaError(msg, __FILE__, __LINE__)

	inline void __GetLastCudaError(const char *ErrorMessage, const char *FileName, const int Line) {
		cudaError_t CUDAError = cudaGetLastError();

		if (cudaSuccess != CUDAError) {
			Debug::Log(
				Debug::LogCritical,
				L"%s(%i) : CUDA error : %s : (%d) %s.",
				CharToWChar(FileName), Line, CharToWChar(ErrorMessage), static_cast<int>(CUDAError),
				CharToWChar(cudaGetErrorString(CUDAError))
			);

			// Make sure we call CUDA Device Reset before exiting
			cudaDeviceReset();
			Debug::Log(Debug::LogNormal, L"Press any key to close...");
			_getch();
			exit(EXIT_FAILURE);
		}
	}

	// --- Beginning of GPU Architecture definitions
	inline int ConvertSMVer2Cores(int major, int minor) {
		// --- Defines for GPU Architecture types (using the SM version to determine
		// --- the # of cores per SM
		typedef struct {
			// --- 0xMm (hexidecimal notation), M = SM Major version,
			int SM;  
			// --- And m = SM minor version
			int Cores;
		} sSMtoCores;

		sSMtoCores nGpuArchCoresPerSM[] = {
			{0x30, 192}, {0x32, 192},
			{0x35, 192}, {0x37, 192},
			{0x50, 128}, {0x52, 128},
			{0x53, 128}, {0x60,  64},
			{0x61, 128}, {0x62, 128},
			{0x70,  64}, {0x72,  64},
			{0x75,  64}, {-1, -1} 
		};

		int index = 0;

		while (nGpuArchCoresPerSM[index].SM != -1) {
			if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
				return nGpuArchCoresPerSM[index].Cores;
			}

			index++;
		}

		// --- If we don't find the values, we default use the previous one
		// --- to run properly
		Debug::Log(
			Debug::LogError,
			L"MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM",
			major, minor, nGpuArchCoresPerSM[index - 1].Cores
		);
		return nGpuArchCoresPerSM[index - 1].Cores;
	}

	//* Returns the best GPU (maximum GFLOPS)
	inline int GetMaxGflopsDeviceId() {
		int CurrentDevice = 0, SMPerMiltiproc = 0;
		int MaxPreformaceDevice = 0;
		int DeviceCount = 0;
		int DevicesProhibited = 0;

		uint64_t MaxComputePerformance = 0;
		cudaDeviceProp DeviceProperties;
		Check(cudaGetDeviceCount(&DeviceCount));

		if (DeviceCount == 0) {
			Debug::Log(
				Debug::LogCritical,
				L"GetMaxGflopsDeviceId(): No devices supporting CUDA."
			);

			Debug::Log(Debug::LogNormal, L"Press any key to close...");
			_getch();
			exit(EXIT_FAILURE);
		}

		// --- Find the best CUDA capable GPU device
		Check(cudaDeviceSynchronize());
		while (CurrentDevice < DeviceCount) {
			cudaGetDeviceProperties(&DeviceProperties, CurrentDevice);

			// --- If this GPU is not running on Compute Mode prohibited,
			// --- then we can add it to the list
			if (DeviceProperties.computeMode != cudaComputeModeProhibited) {
				if (DeviceProperties.major == 9999 && DeviceProperties.minor == 9999) {
					SMPerMiltiproc = 1;
				} else {
					SMPerMiltiproc =
						ConvertSMVer2Cores(DeviceProperties.major, DeviceProperties.minor);
				}

				uint64_t ComputePerformance = (uint64_t)DeviceProperties.multiProcessorCount *
					SMPerMiltiproc * DeviceProperties.clockRate;

				if (ComputePerformance > MaxComputePerformance) {
					MaxComputePerformance = ComputePerformance;
					MaxPreformaceDevice = CurrentDevice;
				}
			} else {
				DevicesProhibited++;
			}

			++CurrentDevice;
		}

		if (DevicesProhibited == DeviceCount) {
			Debug::Log(
				Debug::LogCritical,
				L"GetMaxGflopsDeviceId(): All devices have compute mode prohibited."
			);

			Debug::Log(Debug::LogNormal, L"Press any key to close...");
			_getch();
			exit(EXIT_FAILURE);
		}

		return MaxPreformaceDevice;
	}

	//* Picks the device with highest Gflops/s
	inline int FindCudaDevice() {
		cudaDeviceProp DeviceProperties;
		int DeviceID = 0;

		DeviceID = GetMaxGflopsDeviceId();
		Check(cudaSetDevice(DeviceID));
		Check(cudaGetDeviceProperties(&DeviceProperties, DeviceID));
		Debug::Log(Debug::LogNormal, L"GPU CUDA Device");
		Debug::Log(
			Debug::LogNormal, L"\u2514> GPU Device #%d: '%s' with compute capability %d.%d", DeviceID,
			CharToWChar(DeviceProperties.name), DeviceProperties.major, DeviceProperties.minor
		);

		return DeviceID;
	}
}