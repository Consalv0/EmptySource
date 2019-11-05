
#include "CoreMinimal.h"
#include "Platform/Windows/WindowsInput.h"
#include "Core/Application.h"
#include "Core/Window.h"

#include <SDL.h>

namespace ESource {

	Input * Input::Instance = new WindowsInput();

	WindowsInput * WindowsInput::GetInputInstance() {
		return static_cast<WindowsInput *>(Input::Instance);
	}

	bool WindowsInput::IsKeyStateNative(EScancode KeyCode, int State) {
		if (State == BS_Down) return InputKeyState[KeyCode].FramePressed;
		return InputKeyState[KeyCode].State & State;
	}

	bool WindowsInput::IsMouseStateNative(EMouseButton Button, int State) {
		if (State == BS_Down) return MouseButtonState[Button].FramePressed;
		return MouseButtonState[Button].State & State;
	}

	Vector2 WindowsInput::GetMousePositionNative(bool Clamp) {
		int MouseX, MouseY;
		if (Clamp) {
			SDL_GetMouseState(&MouseX, &MouseY);
		}
		else {
			int WindowX, WindowY;
			SDL_GetGlobalMouseState(&MouseX, &MouseY);
			SDL_GetWindowPosition((SDL_Window *)Application::GetInstance()->GetWindow().GetHandle(), &WindowX, &WindowY);
			MouseX -= WindowX;
			MouseY -= WindowY;
		}
		return Vector2(float(MouseX), float(MouseY));
	}

	float WindowsInput::GetMouseXNative(bool Clamp) {
		int MouseX, MouseY;
		if (Clamp) {
			SDL_GetMouseState(&MouseX, &MouseY);
		}
		else {
			int WindowX, WindowY;
			SDL_GetGlobalMouseState(&MouseX, &MouseY);
			SDL_GetWindowPosition((SDL_Window *)Application::GetInstance()->GetWindow().GetHandle(), &WindowX, &WindowY);
			MouseX -= WindowX;
		}
		return (float)MouseX;
	}

	float WindowsInput::GetMouseYNative(bool Clamp) {
		int MouseX, MouseY;
		if (Clamp) {
			SDL_GetMouseState(&MouseX, &MouseY);
		}
		else {
			int WindowX, WindowY;
			SDL_GetGlobalMouseState(&MouseX, &MouseY);
			SDL_GetWindowPosition((SDL_Window *)Application::GetInstance()->GetWindow().GetHandle(), &WindowX, &WindowY);
			MouseY -= WindowY;
		}
		return (float)MouseY;
	}

}