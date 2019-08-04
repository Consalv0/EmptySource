
#include "Platform/Windows/WindowsWindow.h"
#include "Engine/Application.h"

#include "Utility/LogCore.h"

#include "Graphics/Graphics.h"
#include "Graphics/Bitmap.h"

#include "../External/SDL2/include/SDL.h"

#include "Platform/OpenGL/OpenGLContext.h"

namespace EmptySource {

	int OnCloseWindow(void * Data, SDL_Event * Event) {
		if (Event->type == SDL_QUIT) {
			Application::GetInstance()->ShouldClose();
		}
		return 0;
	}

	void WindowsWindow::Initialize() {

		if (Context != NULL || WindowHandle != NULL) return;

		if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) {
			Debug::Log(Debug::LogCritical, L"Failed to initialize SDL 2.0.9: %ls\n", StringToWString(SDL_GetError()).c_str());
			return;
		}

		if ((WindowHandle = SDL_CreateWindow(
			Name.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, Width, Height,
			SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN | Mode
		)) == NULL) {
			Debug::Log(Debug::LogCritical, L"Window: \"%ls\" could not be initialized: %ls",
				CharToWString(Name.c_str()).c_str(), StringToWString(SDL_GetError()).c_str()
			);
			return;
		}

		Context = std::unique_ptr<OpenGLContext>(new OpenGLContext((SDL_Window *)WindowHandle, 4, 6));

		SDL_AddEventWatch(OnCloseWindow, (void *)this);

		Context->Initialize();
	}

	WindowsWindow::WindowsWindow(const WindowProperties & Properties) {
		Context = NULL;
		WindowHandle = NULL;
		Width = Properties.Width;
		Height = Properties.Height;
		Name = Properties.Title;
		Mode = WindowMode::WM_Windowed;
		Initialize();
	}

	WindowsWindow::~WindowsWindow() {
		Terminate();
		SDL_Quit();
	}

	void WindowsWindow::Resize(const unsigned int & Wth, const unsigned int & Hht) {
		if (Width != Wth || Height != Hht) {
			Width = Wth; Height = Hht;
			SDL_SetWindowSize((SDL_Window *)(WindowHandle), Wth, Hht);
		}
	}

	void WindowsWindow::SetName(const WString & NewName) {
		Name = WStringToString(NewName);
		SDL_SetWindowTitle((SDL_Window *)(WindowHandle), Name.c_str());
	}

	Vector2 WindowsWindow::GetMousePosition(bool Clamp) {
		int MouseX, MouseY;
		if (Clamp) {
			SDL_GetMouseState(&MouseX, &MouseY);
		}
		else {
			int WindowX, WindowY;
			SDL_GetGlobalMouseState(&MouseX, &MouseY);
			SDL_GetWindowPosition((SDL_Window *)(WindowHandle), &WindowX, &WindowY);
			MouseX -= WindowX;
			MouseY -= WindowY;
		}
		return Vector2(float(MouseX), float(MouseY));
	}

	Window * Window::Create(const WindowProperties& Properties) {
		return new WindowsWindow(Properties);
	}

	void WindowsWindow::SetIcon(Bitmap<UCharRGBA>* Icon) {
		SDL_Surface * Surface = SDL_CreateRGBSurfaceFrom(
			(void*)Icon->PointerToValue(),
			Icon->GetWidth(), Icon->GetHeight(),
			32, 4 * Icon->GetWidth(),
			0xF000, 0x0F00, 0x00F0, 0x000F
		);
		SDL_SetWindowIcon(static_cast<SDL_Window *>(WindowHandle), Surface);
		SDL_FreeSurface(Surface);
	}

	void WindowsWindow::EndFrame() {
		Context->SwapBuffers();

		SDL_Event Event;
		while (SDL_PollEvent(&Event)) {
		}
	}

	void WindowsWindow::Terminate() {
		if (IsRunning()) {
#ifdef ES_DEBUG
			Debug::Log(Debug::LogInfo, L"Window: \"%ls\" closed!", GetName().c_str());
#endif // ES_DEBUG
			SDL_DestroyWindow((SDL_Window *)(WindowHandle));
			WindowHandle = NULL;
			SDL_DelEventWatch(OnCloseWindow, (void *)this);
			// SDL_DelEventWatch(OnResizeWindow, this);
		}
	}

	bool WindowsWindow::IsRunning() {
		return WindowHandle != NULL && Context->IsValid();
	}

	WString WindowsWindow::GetName() const {
		return StringToWString(Name);
	}

	float WindowsWindow::GetAspectRatio() const {
		return (float)Width / (float)Height;
	}

	unsigned int WindowsWindow::GetWidth() const {
		return Width;
	}

	unsigned int WindowsWindow::GetHeight() const {
		return Height;
	}

	void * WindowsWindow::GetHandle() const	{
		return WindowHandle;
	}

}
