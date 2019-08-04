
#include "Platform/Windows/WindowsInput.h"
#include "Engine/Application.h"
#include "Engine/Window.h"

#include "../External/SDL2/include/SDL.h"

namespace EmptySource {

	Input * Input::Instance = new WindowsInput();

	bool WindowsInput::IsKeyDownNative(int KeyCode) {
		const Uint8 * Keys = SDL_GetKeyboardState(NULL);
		return Keys[KeyCode];
	}

	bool WindowsInput::IsMouseButtonDownNative(int Button) {
		const Uint32 Flag = SDL_GetMouseState(NULL, NULL);
		return Flag & SDL_BUTTON(Button);
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