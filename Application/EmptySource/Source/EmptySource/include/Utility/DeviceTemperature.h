#pragma once

#ifndef __APPLE__
#include <nvml.h>
#endif
#include "../include/Core.h"

namespace Debug {
	inline int GetDeviceTemperature(const int & DeviceIndex) {
#ifdef WIN32
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
#endif
        
        for(;;) {
            String DeviceTemperature;
            int i = 0;
            std::ifstream inFile("/proc/acpi/thermal_zone/THRM/temperature");
            if (inFile.fail()) {
                return 0;
            }
            while (inFile >> DeviceTemperature) {
                i++;
                if(i == 2)
                    std::cout<<"The Themperature is " << DeviceTemperature << std::endl;
            }
            inFile.close();
        }
        return 0;
	} 
}
