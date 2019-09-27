#pragma once

namespace ESource {

	class DeviceFunctions {
	public:
		virtual ~DeviceFunctions() = default;

		// The device is currently powered by a battery?
		virtual bool IsRunningOnBattery() = 0;

		// Returns the battery life, -1 if no information
		virtual float GetBatteryStatus() = 0;

		virtual float GetDeviceTemperature(const int& DeviceIndex) = 0;

		//* Creates a handler
		static DeviceFunctions * Create();
	};

}