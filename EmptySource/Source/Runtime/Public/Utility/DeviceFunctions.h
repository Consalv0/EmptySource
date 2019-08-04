#pragma once

#include "Engine/Core.h"

namespace EmptySource {

	namespace Debug {
		//* Initialize Nvidia Managment Library or IOKitLib
		bool InitializeDeviceFunctions();

		bool CloseDeviceFunctions();

		float GetDeviceTemperature(const int& DeviceIndex);
	}

}