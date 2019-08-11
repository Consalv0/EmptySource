#pragma once

#include "Core.h"

namespace EmptySource {

	namespace Debug {
		//* Initialize Nvidia Managment Library or IOKitLib
		bool InitializeDeviceFunctions();

		bool CloseDeviceFunctions();

		bool IsRunningOnBattery();

		float GetDeviceTemperature(const int& DeviceIndex);
	}

}