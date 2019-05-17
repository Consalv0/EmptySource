
#include "../include/Core.h"
#include "../include/GLFunctions.h"

// SDL 2.0.9
#include "../External/SDL/include/SDL.h"
#include "../External/SDL/include/SDL_opengl.h"

#include "../include/Window.h"
#include "../include/Bitmap.h"
#include "../include/Math/MathUtility.h"
#include "../include/Math/IntVector2.h"
#include "../include/Math/Vector2.h"
#include "../include/Math/Vector3.h"

int OnCloseWindow(void * Data, SDL_Event * Event) {
	if (Event->type == SDL_QUIT) {
		*static_cast<bool *>(Data) = true;
	}
	return 0;
}

int OnResizeWindow(void * Data, SDL_Event * Event) {
	if (Event->type == SDL_WINDOWEVENT && Event->window.event == SDL_WINDOWEVENT_RESIZED) {
		static_cast<ContextWindow *>(Data)->Resize(Event->window.data1, Event->window.data2);
	}
	return 0;
}

ContextWindow::ContextWindow() {
    Window = NULL;
    Name = "EmptySource Window";
    Width = 1080;
    Height = 720;
    Mode = WindowMode::Windowed;
    FrameCount = 0;
    OnWindowResizedFunc = 0;
}

bool ContextWindow::Create() {
	Window = SDL_CreateWindow(
		Name.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, Width, Height,
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN | Mode
	);
	if (!IsCreated()) {
		Debug::Log(Debug::LogCritical, L"Window: \"%ls\" could not be initialized: %s", CharToWString(Name.c_str()).c_str(), SDL_GetError());
		return false;
	}

	Debug::Log(Debug::LogInfo, L"Window: \"%ls\" initialized!", CharToWString(Name.c_str()).c_str());

	CreateWindowEvents();
	MakeContext();
	InitializeInputs();

	return true;
}

bool ContextWindow::Create(const char * Name, const WindowMode& Mode, const unsigned int& Width, const unsigned int& Height) {
	if (IsCreated()) {
		Debug::Log(Debug::LogWarning, L"Window Already Created!");
		return false;
	}

	this->Width = Width;
	this->Height = Height;
	this->Name = Name;
	this->Mode = Mode;
	this->bShouldClose = !Create();

	return !bShouldClose;
}

void ContextWindow::MakeContext() {
	GLContext = SDL_GL_CreateContext(Window);
}

int ContextWindow::GetWidth() {
    return Width;
}

int ContextWindow::GetHeight() {
    return Height;
}

float ContextWindow::AspectRatio() {
	return (float)Width / (float)Height;
}

void ContextWindow::Resize(const unsigned int& SizeX, const unsigned int& SizeY) {
	if (Width != SizeX || Height != SizeY) {
		OnWindowResized(SizeX, SizeY);
		SDL_SetWindowSize(Window, SizeX, SizeY);
	}
}

void ContextWindow::SetOnResizedEvent(void(*OnWindowResizedFunc)(int Width, int Height)) {
	this->OnWindowResizedFunc = OnWindowResizedFunc;
}

void ContextWindow::OnWindowResized(int Width, int Height) {
	this->Width = Width;
	this->Height = Height;
	if (OnWindowResizedFunc != 0) {
		OnWindowResizedFunc(Width, Height);
	}
}

void ContextWindow::SetWindowName(const WString & NewName) {
    Name = WStringToString(NewName);
	SDL_SetWindowTitle(Window, Name.c_str());
}

WString ContextWindow::GetName() {
    return StringToWString(Name);
}

bool ContextWindow::IsCreated() {
	return Window != NULL;
}

bool ContextWindow::ShouldClose() {
	return bShouldClose;
}

unsigned long long ContextWindow::GetFrameCount() {
    return FrameCount;
}

Vector2 ContextWindow::GetMousePosition(bool Clamp) {
	int MouseX, MouseY;
	if (Clamp) {
		SDL_GetMouseState(&MouseX, &MouseY);
	}
	else {
		int WindowX, WindowY;
		SDL_GetGlobalMouseState(&MouseX, &MouseY);
		SDL_GetWindowPosition(Window, &WindowX, &WindowY);
		MouseX -= WindowX;
		MouseY -= WindowY;
	}
	return Vector2(float(MouseX), float(MouseY));
}

bool ContextWindow::GetKeyDown(unsigned int Key) {
	const Uint8 * Keys = SDL_GetKeyboardState(NULL);
	return Keys[Key];
}

void ContextWindow::ClearWindow() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void ContextWindow::EndOfFrame() {
	SDL_GL_SwapWindow(Window);
    FrameCount++;
}

void ContextWindow::SetIcon(Bitmap<UCharRGBA>* Icon) {
	SDL_Surface * Surface = SDL_CreateRGBSurfaceFrom(
		(void*)Icon->PointerToValue(),
		Icon->GetWidth(), Icon->GetHeight(),
		32, 4 * Icon->GetWidth(),
		0xF000, 0x0F00, 0x00F0, 0x000F
	);
	SDL_SetWindowIcon(Window, Surface);
	SDL_FreeSurface(Surface);
}

void ContextWindow::PollEvents() {
	SDL_Event Event;
	while (SDL_PollEvent(&Event)) {
	}
}

void ContextWindow::InitializeInputs() {
    if (!IsCreated()) {
        Debug::Log(Debug::LogError, L"Unable to set input mode!");
        return;
    }
    
    // glfwSetInputMode(Window, GLFW_STICKY_KEYS, GL_TRUE);
}

void ContextWindow::Terminate() {
    if (IsCreated()) {
        Debug::Log(Debug::LogInfo, L"Window: \"%ls\" closed!", GetName().c_str());
		SDL_DestroyWindow(Window);
		SDL_GL_DeleteContext(GLContext);
		SDL_DelEventWatch(OnCloseWindow, (void *)&bShouldClose);
		SDL_DelEventWatch(OnResizeWindow, this);
		bShouldClose = true;
    }
}

void ContextWindow::CreateWindowEvents() {
	SDL_AddEventWatch(OnCloseWindow, (void *)&bShouldClose);
	SDL_AddEventWatch(OnResizeWindow, this);
}