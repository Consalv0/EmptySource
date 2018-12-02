
#include "..\include\Core.h"
#include "..\include\Window.h"
#include "..\include\Math\Math.h"

ApplicationWindow::ApplicationWindow() {
	Window = NULL;
	Name = "EmptySource Window";
	Width = 1080;
	Height = 720;
	Mode = ES_WINDOW_MODE_WINDOWED;
	FrameCount = 0;
}

int ApplicationWindow::GetWidth() {
	return Width;
}

int ApplicationWindow::GetHeight() {
	return Height;
}

float ApplicationWindow::AspectRatio() {
	return (float)Width / (float)Height;
}

bool ApplicationWindow::Create() {
	GLFWmonitor* PrimaryMonitor = Mode == 1 ? glfwGetPrimaryMonitor() : nullptr;

	Window = glfwCreateWindow(Width, Height, Name, PrimaryMonitor, nullptr);

	wprintf(L"Window: \"%s\" initialized!\n", FChar(Name));

	return false;
}

bool ApplicationWindow::Create(const char * Name, const unsigned int& Mode, const unsigned int& Width, const unsigned int& Height) {
	
	if (IsCreated()) {
		wprintf(L"Error :: Window Already Created!\n");
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
	glfwPollEvents();

	FrameCount++;
}

void ApplicationWindow::InitializeInputs() {
	if (Window == NULL) {
		wprintf(L"Error :: Unable to set input mode!\n");
		return;
	}
	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(Window, GLFW_STICKY_KEYS, GL_TRUE);
}

void ApplicationWindow::Terminate() {
	if (Window != NULL) {
		glfwDestroyWindow(Window);
		wprintf(L"Window: \"%s\" closed!\n", FChar(Name));
	}
}

void ApplicationWindow::OnWindowResized(int Width, int Height) {
	this->Width = Width;
	this->Height = Height;
	glViewport(0, 0, Width, Height);
}