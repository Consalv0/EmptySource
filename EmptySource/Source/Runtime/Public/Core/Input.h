#pragma once

#include "Events/KeyCodes.h"
#include "Math/MathUtility.h"
#include "Math/Vector2.h"

namespace ESource {

	class Input {
	public:
		inline static bool IsKeyDown(Scancode Code) { return Instance->IsKeyStateNative(Code, ButtonState_Down); }

		inline static bool IsKeyPressed(Scancode Code) { return Instance->IsKeyStateNative(Code, ButtonState_Pressed); }

		inline static bool IsKeyReleased(Scancode Code) { return Instance->IsKeyStateNative(Code, ButtonState_Released); }

		inline static bool IsMouseDown(MouseButton Button) { return Instance->IsMouseStateNative(Button, ButtonState_Down); }

		inline static bool IsMousePressed(MouseButton Button) { return Instance->IsMouseStateNative(Button, ButtonState_Pressed); }

		inline static bool IsMouseReleased(MouseButton Button) { return Instance->IsMouseStateNative(Button, ButtonState_Released); }
		
		inline static Vector2 GetMousePosition(bool Clamp = false) { return Instance->GetMousePositionNative(Clamp); }
		
		inline static float GetMouseX(bool Clamp = false) { return Instance->GetMouseXNative(Clamp); }
		
		inline static float GetMouseY(bool Clamp = false) { return Instance->GetMouseYNative(Clamp); }
	
	protected:
		virtual bool IsKeyStateNative(Scancode KeyCode, int State) = 0;

		virtual bool IsMouseStateNative(MouseButton Button, int State) = 0;
		
		virtual Vector2 GetMousePositionNative(bool Clamp) = 0;
		
		virtual float GetMouseXNative(bool Clamp) = 0;
		
		virtual float GetMouseYNative(bool Clamp) = 0;

		static Input * Instance;
	};
}
