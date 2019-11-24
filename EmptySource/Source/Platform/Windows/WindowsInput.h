#pragma once

#include "Core/Input.h"

namespace ESource {

	class WindowsInput : public Input {
	public:
		static WindowsInput * GetInputInstance();

		TArray<DeviceJoystickState> JoystickDeviceState;

		TDictionary<int, TDictionary<EJoystickButton, InputJoystickState>> JoystickButtonState;

		TDictionary<EScancode, InputScancodeState> KeyboardInputState;

		TDictionary<EMouseButton, InputMouseButtonState> MouseInputState;

	protected:
		friend class WindowsWindow;

		virtual bool IsKeyStateNative(EScancode KeyCode, int State) override;

		virtual bool IsMouseStateNative(EMouseButton Button, int State) override;

		virtual bool IsButtonStateNative(int Index, EJoystickButton KeyCode, int State) override;

		virtual float GetAxisNative(int Index, EJoystickAxis Axis) override;

		virtual DeviceJoystickState & GetJoystickStateNative(int Index) override;

		virtual void SendHapticImpulseNative(int Index, int Channel, float Amplitude, int Duration) override;

		virtual TArray<int> GetJoysticksConnected() override;

		virtual void CheckForConnectedJoysticks() override;

		virtual Vector2 GetMousePositionNative(bool Clamp) override;

		virtual float GetMouseXNative(bool Clamp) override;

		virtual float GetMouseYNative(bool Clamp) override;

	};

	int OnSDLWindowInputEvent(void * UserData, SDL_Event * Event);

}

