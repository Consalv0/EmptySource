
#include "EmptySource/include/EmptyHeaders.h"
#include "EmptySource/include/SMath.h"
#include "..\include\SWindow.h"


SWindow::SWindow() {
	Window = NULL;
	Name = "EmptySource Window";
	Width = 1080;
	Height = 720;
	Mode = ES_WINDOW_MODE_WINDOWED;
	FrameCount = 0;
}

int SWindow::GetWidth() {
	return Width;
}

int SWindow::GetHeight() {
	return Height;
}

float SWindow::AspectRatio() {
	return (float)Width / (float)Height;
}

bool SWindow::Create() {
	GLFWmonitor* PrimaryMonitor = Mode == 1 ? glfwGetPrimaryMonitor() : nullptr;

	Window = glfwCreateWindow(Width, Height, Name, PrimaryMonitor, nullptr);

	printf("Window: \"%s\" initialized!\n", Name);

	return false;
}

bool SWindow::Create(const char * Name, const unsigned int& Mode, const unsigned int& Width, const unsigned int& Height) {
	
	if (IsCreated()) {
		printf("Error :: Window Already Created!\n");
		return false;
	}

	this->Width = Width;
	this->Height = Height;
	this->Name = Name;
	this->Mode = Mode;

	return Create();
}

bool SWindow::ShouldClose() {
	return glfwWindowShouldClose(Window);
}

void SWindow::MakeContext() {
	glfwMakeContextCurrent(Window);

	glfwSetWindowUserPointer(Window, this);

	auto WindowResizeFunc = [](GLFWwindow* Handle, int Width, int Height) {
		static_cast<SWindow*>(glfwGetWindowUserPointer(Handle))->OnWindowResized(Width, Height);
	};
	glfwSetWindowSizeCallback(Window, WindowResizeFunc);
}

bool SWindow::IsCreated() {
	return Window != NULL;
}

unsigned long SWindow::GetFrameCount() {
	return FrameCount;
}

FVector2 SWindow::GetMousePosition() {
	double x, y;
	glfwGetCursorPos(Window, &x, &y);
	return FVector2(float(x), float(y));
}

bool SWindow::GetKeyPressed(unsigned int Key) {
	return glfwGetKey(Window, Key) != GLFW_PRESS;
}

void SWindow::EndOfFrame() {
	glfwSwapBuffers(Window);
	glfwPollEvents();

	FrameCount++;
}

void SWindow::InitializeInputs() {
	if (Window == NULL) {
		printf("Error :: Unable to set input mode!\n");
		return;
	}
	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(Window, GLFW_STICKY_KEYS, GL_TRUE);
}

void SWindow::Terminate() {
	if (Window != NULL) {
		glfwDestroyWindow(Window);
		printf("Window: \"%s\" closed!\n", Name);
	}
}

void SWindow::OnWindowResized(int Width, int Height) {
	this->Width = Width;
	this->Height = Height;
	glViewport(0, 0, Width, Height);
}