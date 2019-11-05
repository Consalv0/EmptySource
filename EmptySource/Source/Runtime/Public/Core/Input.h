#pragma once

#include "Events/KeyCodes.h"
#include "Math/MathUtility.h"
#include "Math/Vector2.h"

namespace ESource {

	class Input {
	public:
		inline static bool IsKeyDown(EScancode Code) { return Instance->IsKeyStateNative(Code, BS_Down); }

		inline static bool IsKeyPressed(EScancode Code) { return Instance->IsKeyStateNative(Code, BS_Pressed); }

		inline static bool IsKeyReleased(EScancode Code) { return Instance->IsKeyStateNative(Code, BS_Released); }

		inline static bool IsMouseDown(EMouseButton Button) { return Instance->IsMouseStateNative(Button, BS_Down); }

		inline static bool IsMousePressed(EMouseButton Button) { return Instance->IsMouseStateNative(Button, BS_Pressed); }

		inline static bool IsMouseReleased(EMouseButton Button) { return Instance->IsMouseStateNative(Button, BS_Released); }
		
		inline static Vector2 GetMousePosition(bool Clamp = false) { return Instance->GetMousePositionNative(Clamp); }
		
		inline static float GetMouseX(bool Clamp = false) { return Instance->GetMouseXNative(Clamp); }
		
		inline static float GetMouseY(bool Clamp = false) { return Instance->GetMouseYNative(Clamp); }
	
	protected:
		virtual bool IsKeyStateNative(EScancode KeyCode, int State) = 0;

		virtual bool IsMouseStateNative(EMouseButton Button, int State) = 0;
		
		virtual Vector2 GetMousePositionNative(bool Clamp) = 0;
		
		virtual float GetMouseXNative(bool Clamp) = 0;
		
		virtual float GetMouseYNative(bool Clamp) = 0;

		static Input * Instance;
	};
}
