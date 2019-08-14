#pragma once

#include "Core/Input.h"

namespace EmptySource {

	class WindowsInput : public Input {
	protected:

		virtual bool IsKeyDownNative(int KeyCode) override;

		virtual bool IsMouseButtonDownNative(int Button) override;

		virtual Vector2 GetMousePositionNative(bool Clamp) override;

		virtual float GetMouseXNative(bool Clamp) override;

		virtual float GetMouseYNative(bool Clamp) override;
	};

}

