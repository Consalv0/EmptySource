
#include "..\include\Core.h"
#include "..\include\Graphics.h"
#include "..\include\Window.h"
#include "..\include\Math\Math.h"

ApplicationWindow::ApplicationWindow() {
	Window = NULL;
	Name = "EmptySource Window";
	Width = 1080;
	Height = 720;
	Mode = WindowMode::Windowed;
	FrameCount = 0;
}

int ApplicationWindow::GetWidth() {
	return Width;
}

int ApplicationWindow::GetHeight() {
	return Height;
}

void ApplicationWindow::SetWindowName(const WString & NewName) {
	Name = WStringToString(NewName);
	glfwSetWindowTitle(Window, Name.c_str());
}

WString ApplicationWindow::GetWindowName() {
	return StringToWString(Name);
}

float ApplicationWindow::AspectRatio() {
	return (float)Width / (float)Height;
}

bool ApplicationWindow::Create() {
	GLFWmonitor* PrimaryMonitor = Mode == WindowMode::FullScreen ? glfwGetPrimaryMonitor() : nullptr;

	Window = glfwCreateWindow(Width, Height, Name.c_str(), PrimaryMonitor, nullptr);

	Debug::Log(Debug::LogNormal, L"Window: \"%s\" initialized!", CharToWChar(Name.c_str()));

	return false;
}

bool ApplicationWindow::Create(const char * Name, const WindowMode& Mode, const unsigned int& Width, const unsigned int& Height) {
	
	if (IsCreated()) {
		Debug::Log(Debug::LogWarning, L"Window Already Created!");
		return false;
	}

	this->Width = Width;
	this->Height = Height;
	this->Name = Name;
	this->Mode = Mode;

	return Create();
}

bool ApplicationWindow::ShouldClose() {
	return glfwWindowShouldClose(Window);
}

void ApplicationWindow::MakeContext() {
	glfwMakeContextCurrent(Window);

	glfwSetWindowUserPointer(Window, this);

	auto WindowResizeFunc = [](GLFWwindow* Handle, int Width, int Height) {
		static_cast<ApplicationWindow*>(glfwGetWindowUserPointer(Handle))->OnWindowResized(Width, Height);
	};
	glfwSetWindowSizeCallback(Window, WindowResizeFunc);
}

bool ApplicationWindow::IsCreated() {
	return Window != NULL;
}

unsigned long ApplicationWindow::GetFrameCount() {
	return FrameCount;
}

Vector2 ApplicationWindow::GetMousePosition() {
	double x, y;
	glfwGetCursorPos(Window, &x, &y);
	return Vector2(float(x), float(y));
}

bool ApplicationWindow::GetKeyDown(unsigned int Key) {
	return glfwGetKey(Window, Key) == GLFW_PRESS;
}

void ApplicationWindow::EndOfFrame() {
	glfwSwapBuffers(Window);

	FrameCount++;
}

void ApplicationWindow::PollEvents() {
	glfwPollEvents();
}

void ApplicationWindow::InitializeInputs() {
	if (Window == NULL) {
		Debug::Log(Debug::LogError, L"Unable to set input mode!");
		return;
	}
	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(Window, GLFW_STICKY_KEYS, GL_TRUE);
}

void ApplicationWindow::Terminate() {
	if (Window != NULL) {
		Debug::Log(Debug::LogNormal, L"Window: \"%s\" closed!", GetWindowName().c_str());
		glfwDestroyWindow(Window);
	}
}

void ApplicationWindow::OnWindowResized(int Width, int Height) {
	this->Width = Width;
	this->Height = Height;
	glViewport(0, 0, Width, Height);
}