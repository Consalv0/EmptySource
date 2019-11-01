#pragma once

#include "Core/Input.h"

namespace ESource {

	class WindowsInput : public Input {
	public:
		static WindowsInput * GetInputInstance();

		TDictionary<Scancode, InputScancodeState> InputKeyState;

		TDictionary<MouseButton, InputMouseButtonState> MouseButtonState;

	protected:
		virtual bool IsKeyStateNative(Scancode KeyCode, int State) override;

		virtual bool IsMouseStateNative(MouseButton Button, int State) override;

		virtual Vector2 GetMousePositionNative(bool Clamp) override;

		virtual float GetMouseXNative(bool Clamp) override;

		virtual float GetMouseYNative(bool Clamp) override;
	};

}

