#pragma once

#include "Math/MathUtility.h"
#include "Math/Vector2.h"

namespace ESource {

	class Input {
	public:
		inline static bool IsKeyDown(int KeyCode) { return Instance->IsKeyDownNative(KeyCode); }

		inline static bool IsMouseButtonDown(int Button) { return Instance->IsMouseButtonDownNative(Button); }
		
		inline static Vector2 GetMousePosition(bool Clamp = false) { return Instance->GetMousePositionNative(Clamp); }
		
		inline static float GetMouseX(bool Clamp = false) { return Instance->GetMouseXNative(Clamp); }
		
		inline static float GetMouseY(bool Clamp = false) { return Instance->GetMouseYNative(Clamp); }
	
	protected:
		virtual bool IsKeyDownNative(int KeyCode) = 0;

		virtual bool IsMouseButtonDownNative(int Button) = 0;
		
		virtual Vector2 GetMousePositionNative(bool Clamp) = 0;
		
		virtual float GetMouseXNative(bool Clamp) = 0;
		
		virtual float GetMouseYNative(bool Clamp) = 0;

	private:
		
		static Input * Instance;
	};
}
