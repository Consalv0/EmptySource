#pragma once

#include "Core/Input.h"

namespace ESource {

	class WindowsInput : public Input {
	public:
		static WindowsInput * GetInputInstance();

		TDictionary<EScancode, InputScancodeState> InputKeyState;

		TDictionary<EMouseButton, InputMouseButtonState> MouseButtonState;

	protected:
		virtual bool IsKeyStateNative(EScancode KeyCode, int State) override;

		virtual bool IsMouseStateNative(EMouseButton Button, int State) override;

		virtual Vector2 GetMousePositionNative(bool Clamp) override;

		virtual float GetMouseXNative(bool Clamp) override;

		virtual float GetMouseYNative(bool Clamp) override;
	};

}

