#pragma once

#include "../../include/Core.h"

namespace EmptySource {

	namespace Debug {
		//* Initialize Nvidia Managment Library or IOKitLib
		bool InitializeDeviceFunctions();

		bool CloseDeviceFunctions();

		float GetDeviceTemperature(const int& DeviceIndex);
	}

}