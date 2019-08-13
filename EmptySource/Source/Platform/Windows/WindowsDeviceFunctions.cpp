
#include "CoreMinimal.h"
#include "Platform/Windows/WindowsDeviceFunctions.h"

#ifdef ES_PLATFORM_NVML
#include <nvml.h>
#endif

namespace EmptySource {

	WindowsDeviceFunctions::WindowsDeviceFunctions() {
#ifdef ES_PLATFORM_NVML
		nvmlReturn_t DeviceResult = nvmlInit();

		if (NVML_SUCCESS != DeviceResult)
			LOG_CORE_ERROR("Failed to initialize NVML: {}", NString(nvmlErrorString(DeviceResult)));
#endif
	}

	WindowsDeviceFunctions::~WindowsDeviceFunctions() {
#ifdef ES_PLATFORM_NVML
		nvmlReturn_t DeviceResult = nvmlShutdown();

		if (NVML_SUCCESS != DeviceResult)
			LOG_CORE_ERROR("Failed to shutdown NVML: {}", NString(nvmlErrorString(DeviceResult)));
#endif
	}

	bool WindowsDeviceFunctions::IsRunningOnBattery() {
		SYSTEM_POWER_STATUS Status;
		GetSystemPowerStatus(&Status);

		switch (Status.ACLineStatus) {
			case 0://	"Offline"
			case 255://	"Unknown status"
				return true;
			case 1://	"Online"
			default:
				return false;
		}
	}

	float WindowsDeviceFunctions::GetBatteryStatus() {
		SYSTEM_POWER_STATUS Status;
		GetSystemPowerStatus(&Status);

		switch (Status.BatteryFlag) {
			case 128://	"No system battery"
			case 255://	"Unknown status-unable to read the battery flag information"
			default:
				return -1.F;
		}

		return Status.BatteryLifePercent;
	}

	float WindowsDeviceFunctions::GetDeviceTemperature(const int & DeviceIndex) {
		unsigned int DeviceTemperature = 0;
#ifdef ES_PLATFORM_NVML
		nvmlDevice_t Device;
		nvmlReturn_t DeviceResult;

		DeviceResult = nvmlDeviceGetHandleByIndex(DeviceIndex, &Device);
		if (NVML_SUCCESS != DeviceResult) {
			LOG_CORE_ERROR("Failed to get handle of NVIDIA device {0:d}: {1}", DeviceIndex, NString(nvmlErrorString(DeviceResult)));
			return (float)DeviceTemperature;
		}

		DeviceResult = nvmlDeviceGetTemperature(Device, NVML_TEMPERATURE_GPU, &DeviceTemperature);
		if (NVML_SUCCESS != DeviceResult)
			LOG_CORE_ERROR("Failed to get temperature of NVIDIA device {0:d}: {1}", DeviceIndex, NString(nvmlErrorString(DeviceResult)));
#endif

		return (float)DeviceTemperature;
	};

	DeviceFunctions * DeviceFunctions::Create() {
		return new WindowsDeviceFunctions();
	}

}