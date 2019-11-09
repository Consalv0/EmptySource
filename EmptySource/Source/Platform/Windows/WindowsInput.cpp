
#include "CoreMinimal.h"
#include "Utility/TextFormatting.h"

#include "Platform/Windows/WindowsWindow.h"
#include "Core/Application.h"
#include "Core/Window.h"

#include <SDL_events.h>
#include "Platform/Windows/WindowsInput.h"

#include <SDL.h>

namespace ESource {

	Input * Input::Instance = new WindowsInput();

	WindowsInput * WindowsInput::GetInputInstance() {
		return static_cast<WindowsInput *>(Input::Instance);
	}

	bool WindowsInput::IsKeyStateNative(EScancode KeyCode, int State) {
		if (State == BS_Down) return KeyboardInputState[KeyCode].FramePressed;
		return KeyboardInputState[KeyCode].State & State;
	}

	bool WindowsInput::IsMouseStateNative(EMouseButton Button, int State) {
		if (State == BS_Down) return MouseInputState[Button].FramePressed;
		return MouseInputState[Button].State & State;
	}

	bool WindowsInput::IsButtonStateNative(int Index, EJoystickButton Button, int State) {
		if (Index == -1) {
			TArray<int> Indices;
			Indices = GetJoysticksConnected();
			if (Indices.empty()) Index = 0;
			else Index = Indices[0];
		}
		if (State == BS_Down) return JoystickButtonState[Index][Button].FramePressed;
		return JoystickButtonState[Index][Button].State & State;
	}

	float WindowsInput::GetAxisNative(int Index, EJoystickAxis Axis) {
		if (Index == -1) {
			TArray<int> Indices;
			Indices = GetJoysticksConnected();
			if (Indices.empty()) Index = 0;
			else Index = Indices[0];
		}
		DeviceJoystickState & Joystick = GetJoystickStateNative(Index);
		SDL_GameController * GController = SDL_GameControllerFromInstanceID(Joystick.InstanceID);
		return SDL_GameControllerGetAxis(GController, (SDL_GameControllerAxis)Axis) / 32768.F;
		return 0.F;
	}

	DeviceJoystickState & WindowsInput::GetJoystickStateNative(int Index) {
		if (Index >= JoystickDeviceState.size()) JoystickDeviceState.resize(Index + 1);
		return JoystickDeviceState[Index];
	}

	TArray<int> WindowsInput::GetJoysticksConnected() {
		TArray<int> Indices; int Count = 0;
		for (auto & DeviceState : JoystickDeviceState) {
			if (DeviceState.bConnected) Indices.push_back(Count);
			Count++;
		}
		
		return Indices;
	}

	void WindowsInput::CheckForConnectedJoysticks() {
		for (int i = 0; i < SDL_NumJoysticks(); i++) {
			bool IsGameController = SDL_IsGameController(i);
			if (IsGameController) {
				SDL_GameController * GameController = SDL_GameControllerOpen(i);
				auto & DevicesState = WindowsInput::GetInputInstance()->JoystickDeviceState;
				IName DeviceName = IName(Text::NarrowToWide(SDL_JoystickNameForIndex(i)), SDL_JoystickInstanceID(SDL_GameControllerGetJoystick(GameController)));
				DeviceJoystickState * JoystickDeviceState = NULL;
				for (auto & State : DevicesState) {
					if (State.Name.GetDisplayName() == DeviceName.GetDisplayName()) {
						if (State.bConnected) return;
						JoystickDeviceState = &State;
						break;
					}
				}
				if (JoystickDeviceState == NULL) {
					for (auto & State : DevicesState) {
						if (State.bConnected == false) {
							JoystickDeviceState = &State;
							break;
						}
					}
				}
				if (JoystickDeviceState == NULL) {
					DevicesState.push_back(DeviceJoystickState());
					JoystickDeviceState = &DevicesState[WindowsInput::GetInputInstance()->JoystickDeviceState.size() - 1];
				}
				JoystickDeviceState->InstanceID = SDL_JoystickInstanceID(SDL_GameControllerGetJoystick(GameController));
				JoystickDeviceState->Name = DeviceName;
				JoystickDeviceState->bConnected = true;
			}
		}
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

	int OnSDLWindowInputEvent(void * UserData, SDL_Event * Event) {
		auto & KeyState = WindowsInput::GetInputInstance()->KeyboardInputState[(EScancode)Event->key.keysym.scancode];
		auto & MouseState = WindowsInput::GetInputInstance()->MouseInputState[(EMouseButton)Event->button.button];
		WindowsWindow& Data = *(WindowsWindow*)UserData;
		static int32_t MouseButtonPressedCount[255] = {
			(int32_t)-1, (int32_t)-1, (int32_t)-1, (int32_t)-1, (int32_t)-1,
			(int32_t)-1, (int32_t)-1, (int32_t)-1, (int32_t)-1, (int32_t)-1
		};

		if (Event->type == SDL_WINDOWEVENT) {
			if (Event->window.windowID != SDL_GetWindowID((SDL_Window*)Data.GetHandle()))
				return 0;
		}

		switch (Event->type) {
		case SDL_WINDOWEVENT: {
			if (Event->window.event == SDL_WINDOWEVENT_RESIZED || Event->window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
				Data.Resize(Event->window.data1, Event->window.data2);
				break;
			}
			if (Event->window.event == SDL_WINDOWEVENT_CLOSE) {
				WindowCloseEvent WinEvent;
				Data.WindowEventCallback(WinEvent);
				break;
			}
			if (Event->window.event == SDL_WINDOWEVENT_FOCUS_GAINED) {
				WindowGainFocusEvent WinEvent;
				Data.WindowEventCallback(WinEvent);
				break;
			}
			if (Event->window.event == SDL_WINDOWEVENT_FOCUS_LOST) {
				WindowLostFocusEvent WinEvent;
				Data.WindowEventCallback(WinEvent);
				break;
			}
		}

		case SDL_JOYDEVICEADDED:
		case SDL_CONTROLLERDEVICEADDED: {
			bool IsGameController = SDL_IsGameController(Event->jdevice.which);
			if (IsGameController) {
				SDL_GameController * GameController = SDL_GameControllerOpen(Event->jdevice.which);
				auto & DevicesState = WindowsInput::GetInputInstance()->JoystickDeviceState;
				IName DeviceName = IName(
					Text::NarrowToWide(SDL_JoystickNameForIndex(Event->jdevice.which)),
					SDL_JoystickInstanceID(SDL_GameControllerGetJoystick(GameController))
				);
				DeviceJoystickState * JoystickDeviceState = NULL;
				for (auto & State : DevicesState) {
					if (State.Name.GetDisplayName() == DeviceName.GetDisplayName()) {
						if (State.bConnected) return 0;
						JoystickDeviceState = &State;
						break;
					}
				}
				if (JoystickDeviceState == NULL) {
					for (auto & State : DevicesState) {
						if (State.bConnected == false) {
							JoystickDeviceState = &State;
							break;
						}
					}
				}
				if (JoystickDeviceState == NULL) {
					DevicesState.push_back(DeviceJoystickState());
					JoystickDeviceState = &DevicesState[WindowsInput::GetInputInstance()->JoystickDeviceState.size() - 1];
				}
				JoystickDeviceState->InstanceID = SDL_JoystickInstanceID(SDL_GameControllerGetJoystick(GameController));
				JoystickDeviceState->Name = DeviceName;
				JoystickDeviceState->bConnected = true;

				LOG_CORE_INFO(L"Device {} Opened", JoystickDeviceState->Name.GetInstanceName());

				JoystickConnectionEvent InEvent(
					JoystickDeviceState->InstanceID, 1
				);
				Data.InputEventCallback(InEvent);
			}
			break;
		}

		case SDL_JOYDEVICEREMOVED:
		case SDL_CONTROLLERDEVICEREMOVED: {
			SDL_Joystick * Joystick = SDL_JoystickFromInstanceID(Event->jdevice.which);

			if (SDL_JoystickGetAttached(Joystick)) {
				DeviceJoystickState * JoystickDeviceState = NULL;
				IName DeviceName = IName(Text::NarrowToWide(SDL_JoystickName(Joystick)), SDL_JoystickInstanceID(Joystick));
				for (auto & State : WindowsInput::GetInputInstance()->JoystickDeviceState) {
					if (State.Name.GetID() == DeviceName.GetID()) {
						JoystickDeviceState = &State;
						break;
					}
				}

				if (JoystickDeviceState == NULL) break;
				JoystickDeviceState->bConnected = false;

				LOG_CORE_INFO(L"Device {} Closed", JoystickDeviceState->Name.GetInstanceName());

				JoystickConnectionEvent InEvent(
					JoystickDeviceState->InstanceID, 0
				);
				Data.InputEventCallback(InEvent);

				SDL_JoystickClose(Joystick);
			}
			break;
		}

		case SDL_JOYBUTTONUP:
		case SDL_CONTROLLERBUTTONUP: {
			SDL_Joystick * Joystick = SDL_JoystickFromInstanceID(Event->jbutton.which);
			if (SDL_JoystickGetAttached(Joystick)) {
				DeviceJoystickState * JoystickDeviceState = NULL;
				IName DeviceName = IName(Text::NarrowToWide(SDL_JoystickName(Joystick)), SDL_JoystickInstanceID(Joystick));
				int Index = 0;
				for (auto & State : WindowsInput::GetInputInstance()->JoystickDeviceState) {
					if (State.Name.GetID() == DeviceName.GetID()) {
						JoystickDeviceState = &State;
						break;
					}
					Index++;
				}
				if (JoystickDeviceState == NULL) break;

				auto & JoyButtonState = WindowsInput::GetInputInstance()->JoystickButtonState[Index][(EJoystickButton)Event->jbutton.button];
				JoyButtonState.State = BS_Released;
				JoyButtonState.FramePressed = 0;

				JoystickButtonReleasedEvent InEvent(
					JoystickDeviceState->InstanceID, (EJoystickButton)Event->cbutton.button
				);
				Data.InputEventCallback(InEvent);
			}
			break;
		}

		case SDL_JOYBUTTONDOWN:
		case SDL_CONTROLLERBUTTONDOWN: {
			SDL_Joystick * Joystick = SDL_JoystickFromInstanceID(Event->jbutton.which);
			if (SDL_JoystickGetAttached(Joystick)) {
				DeviceJoystickState * JoystickDeviceState = NULL;
				IName DeviceName = IName(Text::NarrowToWide(SDL_JoystickName(Joystick)), SDL_JoystickInstanceID(Joystick));
				int Index = 0;
				for (auto & State : WindowsInput::GetInputInstance()->JoystickDeviceState) {
					if (State.Name.GetID() == DeviceName.GetID()) {
						JoystickDeviceState = &State;
						break;
					}
					Index++;
				}
				if (JoystickDeviceState == NULL) break;

				auto & JoyButtonState = WindowsInput::GetInputInstance()->JoystickButtonState[Index][(EJoystickButton)Event->jbutton.button];
				JoyButtonState.State = BS_Pressed;

				JoystickButtonPressedEvent InEvent(
					JoystickDeviceState->InstanceID, (EJoystickButton)Event->cbutton.button
				);
				Data.InputEventCallback(InEvent);
			}
			break;
		}

		case SDL_JOYAXISMOTION:
		case SDL_CONTROLLERAXISMOTION: {
			SDL_Joystick * Joystick = SDL_JoystickFromInstanceID(Event->jdevice.which);

			if (SDL_JoystickGetAttached(Joystick)) {
				DeviceJoystickState * JoystickDeviceState = NULL;
				IName DeviceName = IName(Text::NarrowToWide(SDL_JoystickName(Joystick)), SDL_JoystickInstanceID(Joystick));
				for (auto & State : WindowsInput::GetInputInstance()->JoystickDeviceState) {
					if (State.Name.GetID() == DeviceName.GetID()) {
						JoystickDeviceState = &State;
						break;
					}
				}
				if (JoystickDeviceState == NULL) break;

				JoystickAxisEvent InEvent(
					JoystickDeviceState->InstanceID, (EJoystickAxis)Event->caxis.axis, Event->caxis.value / 32768.F
				);
				Data.InputEventCallback(InEvent);
			}
			break;
		}

		case SDL_KEYDOWN: {
			if (Event->key.repeat == 0)
				KeyState.State = BS_Pressed | BS_Typed;
			else
				KeyState.State = BS_Typed;
			KeyState.TypeRepeticions = Event->key.repeat;

			KeyPressedEvent InEvent(
				Event->key.keysym.scancode,
				(SDL_GetModState() & KMOD_SHIFT) != 0,
				(SDL_GetModState() & KMOD_CTRL) != 0,
				(SDL_GetModState() & KMOD_ALT) != 0,
				(SDL_GetModState() & KMOD_GUI) != 0,
				Event->key.repeat
			);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_KEYUP: {
			KeyState.State = BS_Released;
			KeyState.FramePressed = 0;
			KeyState.TypeRepeticions = 0;

			KeyReleasedEvent InEvent(
				Event->key.keysym.scancode,
				(SDL_GetModState() & KMOD_SHIFT) != 0,
				(SDL_GetModState() & KMOD_CTRL) != 0,
				(SDL_GetModState() & KMOD_ALT) != 0,
				(SDL_GetModState() & KMOD_GUI) != 0
			);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_TEXTINPUT: {
			KeyTypedEvent InEvent(Event->text.text);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_MOUSEBUTTONDOWN: {
			MouseState.State = BS_Pressed;
			MouseState.Clicks = Event->button.clicks;

			MouseButtonPressedCount[Event->button.button]++;
			MouseButtonPressedEvent InEvent(Event->button.button, Event->button.clicks == 2, MouseButtonPressedCount[Event->button.button]);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_MOUSEBUTTONUP: {
			MouseState.State = BS_Released;
			MouseState.FramePressed = 0;
			MouseState.Clicks = 0;

			MouseButtonPressedCount[Event->button.button] = -1;
			MouseButtonReleasedEvent InEvent(Event->button.button);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_MOUSEMOTION: {
			MouseMovedEvent InEvent((float)Event->motion.x, (float)Event->motion.y, (float)Event->motion.xrel, (float)Event->motion.yrel);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_MOUSEWHEEL: {
			MouseScrolledEvent InEvent(
				(float)Event->wheel.x, (float)Event->wheel.y,
				Event->wheel.direction == SDL_MOUSEWHEEL_FLIPPED
			);
			Data.InputEventCallback(InEvent);
			break;
		}
		}

		return 0;
	}

}