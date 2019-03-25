#pragma once

#include "../../include/Core.h"

namespace Debug {
    bool InitializeDeviceFunctions();
    bool CloseDeviceFunctions();
    float GetDeviceTemperature(const int& DeviceIndex);
}
