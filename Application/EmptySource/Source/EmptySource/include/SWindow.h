#pragma once

// Window modes
#define ES_WINDOW_MODE_WINDOWED             0
#define ES_WINDOW_MODE_WINDOWEDFULLSCREEN   1
#define ES_WINDOW_MODE_FULLSCREEN           2

/*
** Used to cointain the properties and functions of a glfw window
*/
struct SWindow {
private:
	const char* Name;
	unsigned int Width = 1080;
	unsigned int Height = 720;
	unsigned int Mode = ES_WINDOW_MODE_WINDOWED;

	//* Event that resizes this window
	void OnWindowResized(int Width, int Height);

public:
	SWindow();
	
	//* Window Window
	struct GLFWwindow* Window;
	
	//* Creates the main Rendering Window
	bool Create();
	//* Creates the main Rendering Window with a Name, Width and Height
	bool Create(const char * Name, const unsigned int& Mode, const unsigned int& Width, const unsigned int& Height);

	//* Wrapper for glfwShouldClose, asks if window should be closed
	bool ShouldClose();

	//* Make context in GLFW for this window
	void MakeContext();

	//* Initialize inputs in this window
	void InitializeInputs();

	//* Terminates this window
	void Destroy();
};