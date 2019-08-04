// 
// #include "Engine/Core.h"
// #include "Graphics/GLFunctions.h"
// 
// // SDL 2.0.9
// #include "../External/SDL2/include/SDL.h"
// 
// #include "Engine/Window.h"
// #include "Graphics/Bitmap.h"
// #include "Math/MathUtility.h"
// #include "Math/IntVector2.h"
// #include "Math/Vector2.h"
// #include "Math/Vector3.h"
// 
// namespace EmptySource {
// 
// 	int OnCloseWindow(void * Data, SDL_Event * Event) {
// 		if (Event->type == SDL_QUIT) {
// 			*static_cast<bool *>(Data) = true;
// 		}
// 		return 0;
// 	}
// 
// 	int OnResizeWindow(void * Data, SDL_Event * Event) {
// 		if (Event->type == SDL_WINDOWEVENT && Event->window.event == SDL_WINDOWEVENT_RESIZED) {
// 			static_cast<Window *>(Data)->Resize(Event->window.data1, Event->window.data2);
// 		}
// 		return 0;
// 	}
// 
// 	bool Window::Create() {
// 		}
// 		Debug::Log(Debug::LogInfo, L"Window: \"%ls\" initialized!", CharToWString(Name.c_str()).c_str());
// 
// 		CreateWindowEvents();
// 		MakeContext();
// 		InitializeInputs();
// 
// 		return true;
// 	}
// 
// 	void Window::SetOnResizedEvent(void(*OnWindowResizedFunc)(int Width, int Height)) {
// 		this->OnWindowResizedFunc = OnWindowResizedFunc;
// 	}
// 
// 	unsigned long long Window::GetFrameCount() {
// 		return FrameCount;
// 	}
// 
// 	bool Window::GetKeyDown(unsigned int Key) {
// 		const Uint8 * Keys = SDL_GetKeyboardState(NULL);
// 		return Keys[Key];
// 	}
// 
// 	void Window::PollEvents() {
// 		SDL_Event Event;
// 		while (SDL_PollEvent(&Event)) {
// 		}
// 	}
// 
// 	void Window::InitializeInputs() {
// 		if (!IsCreated()) {
// 			Debug::Log(Debug::LogError, L"Unable to set input mode!");
// 			return;
// 		}
// 
// 		// glfwSetInputMode(Window, GLFW_STICKY_KEYS, GL_TRUE);
// 	}
// 
// 	void Window::CreateWindowEvents() {
// 		SDL_AddEventWatch(OnCloseWindow, (void *)&bShouldClose);
// 		SDL_AddEventWatch(OnResizeWindow, this);
// 	}
// 
// 	float WindowProperties::AspectRatio() const {
// 		return (float)Width / (float)Height;
// 	}
// }