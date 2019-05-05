#pragma once

// Include GLFW
// Library to make crossplataform input and window creation
#include "External/GLFW/glfw3.h"

#include "../include/Graphics.h"
#include "../include/Text.h"

enum WindowMode {
    Windowed = 0,
    FullScreen = 1
};

template <typename T>
class Bitmap;

	/*
 ** Cointaina the properties and functions of a GLFW window
 */
struct ApplicationWindow {
private:
    //* GLFW Window
    struct GLFWwindow * Window;
    
    //* External on resized function
    void(*OnWindowResizedFunc)(int Width, int Height);
    
    //* Name displayed in header window
    String Name;
    unsigned int Width = 1080;
    unsigned int Height = 720;
    WindowMode Mode = WindowMode::Windowed;
    
    //* Frame Count
    unsigned long FrameCount;
    
    //* Event callback that resizes this window
    void OnWindowResized(int Width, int Height);
    
    //* Creates the main Rendering Window
    bool Create();
    
public:
    
    ApplicationWindow();
    
    //* Get the width in pixels of the window
    int GetWidth();
    
    //* Get the height in pixels of the window
    int GetHeight();
    
    //* Rename the window title
    void SetWindowName(const WString & NewName);
    
    //* Get the window title name
    WString GetWindowName();
    
    //* Get the aspect of width divided by height in pixels of the window
    float AspectRatio();
    
    //* Creates a Window with a Name, Width and Height
    bool Create(const char * Name, const WindowMode& Mode, const unsigned int& Width, const unsigned int& Height);
    
    //* Wrapper for glfwShouldClose, asks if window should be closed
    bool ShouldClose();
    
    //* Make context in GLFW for this window
    void MakeContext();
    
    //* Returns true if window has been created
    bool IsCreated();
    
    //* Total frames drawed since the creation of this window
    unsigned long GetFrameCount();
    
    //* Get mouse position in screen coordinates relative to the upper left position of this window
    struct Vector2 GetMousePosition();
    
    //* Get key pressed
    bool GetKeyDown(unsigned int Key);
    
    //* Window clear events
    void ClearWindow();
    
    //* Window update frame
    void EndOfFrame();
    
	//* Sets the window icon
	void SetIcon(class Bitmap<UCharRGBA> * Icon);

	//* Sets the window icon using the resources.h
#ifdef WIN32
	void SetIcon(const int & IconResource);
#endif

    //* Window update events
    void PollEvents();
    
    //* Initialize inputs in this window
    void InitializeInputs();
    
    //* Terminates this window
    void Terminate();
    
    //* Set on resized event
    void SetOnResizedEvent(void(*OnWindowResizedFunc)(int Width, int Height));
};
