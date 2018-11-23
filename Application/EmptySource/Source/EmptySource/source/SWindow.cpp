
#include "EmptySource/include/EmptyHeaders.h"
#include "..\include\SWindow.h"

SWindow::SWindow() {
	Window = NULL;
	Name = "EmptySource Window";
	Width = 1080;
	Height = 720;
	Mode = ES_WINDOW_MODE_WINDOWED;
}

bool SWindow::Create() {
	if (!glfwInit()) {
		printf("Error :: Failed to initialize GLFW\n");
		return false;
	} 

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	Window = glfwCreateWindow(Width, Height, Name, nullptr, nullptr);

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