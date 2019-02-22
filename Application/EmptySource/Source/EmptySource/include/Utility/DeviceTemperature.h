#pragma once

#include <nvml.h>
#include "EmptySource\include\Core.h"

namespace Debug {
	inline int GetDeviceTemperature(const int & DeviceIndex) {
		nvmlDevice_t Device;
		nvmlReturn_t DeviceResult;
		unsigned int DeviceTemperature;

		DeviceResult = nvmlDeviceGetHandleByIndex(DeviceIndex, &Device);
		if (NVML_SUCCESS != DeviceResult) {
			Debug::Log(Debug::LogError, L"NVML :: Failed to get handle for device %i: %s", DeviceIndex, CharToWChar(nvmlErrorString(DeviceResult)));
			return 0;
		}

		DeviceResult = nvmlDeviceGetTemperature(Device, NVML_TEMPERATURE_GPU, &DeviceTemperature);
		if (NVML_SUCCESS != DeviceResult) {
			Debug::Log(Debug::LogError, L"NVML :: Failed to get temperature of device %i: %s", DeviceIndex, CharToWChar(nvmlErrorString(DeviceResult)));
		}
		return DeviceTemperature;

		return 0;
	} 
}