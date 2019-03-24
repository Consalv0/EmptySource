#pragma once

#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#elif WIN32
#include <nvml.h>
#endif
#include "../../include/Core.h"

namespace Debug {
    bool InitializeDeviceFunctions();
    bool CloseDeviceFunctions();
    float GetDeviceTemperature(const int& DeviceIndex);
}
