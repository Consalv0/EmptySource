#pragma once

#include "Events/KeyCodes.h"
#include "Math/MathUtility.h"
#include "Math/Vector2.h"

namespace ESource {

	struct DeviceJoystickState {
		IName Name;
		int InstanceID;
		bool bConnected;
		int Mapping;

		DeviceJoystickState() : Name(L"", 0), InstanceID(0), bConnected(false), Mapping(0) {};
	};

	class Input {
	public:
		inline static bool IsKeyDown(EScancode Code) { return Instance->IsKeyStateNative(Code, BS_Down); }

		inline static bool IsKeyPressed(EScancode Code) { return Instance->IsKeyStateNative(Code, BS_Pressed); }

		inline static bool IsKeyReleased(EScancode Code) { return Instance->IsKeyStateNative(Code, BS_Released); }

		inline static bool IsMouseDown(EMouseButton Button) { return Instance->IsMouseStateNative(Button, BS_Down); }

		inline static bool IsMousePressed(EMouseButton Button) { return Instance->IsMouseStateNative(Button, BS_Pressed); }

		inline static bool IsMouseReleased(EMouseButton Button) { return Instance->IsMouseStateNative(Button, BS_Released); }

		inline static float GetAxis(int Index, EJoystickAxis Axis) { return Instance->GetAxisNative(Index, Axis); }

		inline static bool IsButtonDown(int Index, EJoystickButton Code) { return Instance->IsButtonStateNative(Index, Code, BS_Down); }

		inline static bool IsButtonPressed(int Index, EJoystickButton Code) { return Instance->IsButtonStateNative(Index, Code, BS_Pressed); }

		inline static bool IsButtonReleased(int Index, EJoystickButton Code) { return Instance->IsButtonStateNative(Index, Code, BS_Released); }

		inline static bool IsJoystickConnected(int Index) { return Instance->GetJoystickStateNative(Index).bConnected; };

		inline static const DeviceJoystickState & GetJoystickState(int Index) { return Instance->GetJoystickStateNative(Index); };
		
		inline static Vector2 GetMousePosition(bool Clamp = false) { return Instance->GetMousePositionNative(Clamp); }
		
		inline static float GetMouseX(bool Clamp = false) { return Instance->GetMouseXNative(Clamp); }
		
		inline static float GetMouseY(bool Clamp = false) { return Instance->GetMouseYNative(Clamp); }
	
	protected:
		virtual bool IsKeyStateNative(EScancode KeyCode, int State) = 0;

		virtual bool IsMouseStateNative(EMouseButton Button, int State) = 0;

		virtual bool IsButtonStateNative(int Index, EJoystickButton KeyCode, int State) = 0;

		virtual float GetAxisNative(int Index, EJoystickAxis Axis) = 0;

		virtual DeviceJoystickState & GetJoystickStateNative(int Index) = 0;

		virtual TArray<int> GetJoysticksConnected() = 0;

		virtual void CheckForConnectedJoysticks() = 0;
		
		virtual Vector2 GetMousePositionNative(bool Clamp) = 0;
		
		virtual float GetMouseXNative(bool Clamp) = 0;
		
		virtual float GetMouseYNative(bool Clamp) = 0;

		static Input * Instance;
	};
}
