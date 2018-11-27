
#include "EmptySource/include/EmptyHeaders.h"
#include "..\include\SWindow.h"

SWindow::SWindow() {
	Window = NULL;
	Name = "EmptySource Window";
	Width = 1080;
	Height = 720;
	Mode = ES_WINDOW_MODE_WINDOWED;
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
	GLFWmonitor* PrimaryMonitor = glfwGetPrimaryMonitor();

	Window = glfwCreateWindow(Width, Height, Name, Mode == 1 ? PrimaryMonitor : nullptr, nullptr);

	printf("Window: \"%s\" initialized!\n", Name);

	return false;
}

bool SWindow::Create(const char * Name, const unsigned int& Mode, const unsigned int& Width, const unsigned int& Height) {
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

void SWindow::InitializeInputs() {
	if (Window == NULL) {
		printf("Error :: Unable to set input mode!\n");
		return;
	}
	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(Window, GLFW_STICKY_KEYS, GL_TRUE);
}

void SWindow::Destroy() {
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