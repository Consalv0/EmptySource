#pragma once

// Window modes
#define ES_WINDOW_MODE_WINDOWED             0
#define ES_WINDOW_MODE_FULLSCREEN           1

/*
** Cointaina the properties and functions of a GLFW window
*/
struct SWindow {
private:
	//* GLFW Window
	struct GLFWwindow* Window;

	//* Name displayed in window
	const char* Name;
	unsigned int Width = 1080;
	unsigned int Height = 720;
	unsigned int Mode = ES_WINDOW_MODE_WINDOWED;

	//* Frame Count
	unsigned long FrameCount;

	//* Event callback that resizes this window
	void OnWindowResized(int Width, int Height);

	//* Creates the main Rendering Window
	bool Create();

public:

	SWindow();

	//* Get the width in pixels of the window
	int GetWidth();
	//* Get the height in pixels of the window
	int GetHeight();

	//* Get the aspect of width divided by height in pixels of the window
	float AspectRatio();

	//* Creates the main Rendering Window with a Name, Width and Height
	bool Create(const char * Name, const unsigned int& Mode, const unsigned int& Width, const unsigned int& Height);

	//* Wrapper for glfwShouldClose, asks if window should be closed
	bool ShouldClose();

	//* Make context in GLFW for this window
	void MakeContext();

	//* Returns true if window has been created
	bool IsCreated();

	//* Total frames drawed
	unsigned long GetFrameCount();

	//* Get mouse position respect this window
	struct FVector2 GetMousePosition();

	//* Get key pressed
	bool GetKeyPressed(unsigned int Key);

	//* Window end of frame call
	void EndOfFrame();

	//* Initialize inputs in this window
	void InitializeInputs();

	//* Terminates this window
	void Terminate();
};