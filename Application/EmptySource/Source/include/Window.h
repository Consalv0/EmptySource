#pragma once

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
struct ContextWindow {
private:
    //* Window Handle
    struct SDL_Window * Window;
	void * GLContext;
    
    //* External on resized function
    void(*OnWindowResizedFunc)(int Width, int Height);

    //* Name displayed in header window
    String Name;
    unsigned int Width = 1080;
    unsigned int Height = 720;
    WindowMode Mode = WindowMode::Windowed;
    
    //* Frame Count
    unsigned long long FrameCount;

	//* The window should close
	bool bShouldClose;
    
    //* Event callback that resizes this window
    void OnWindowResized(int Width, int Height);

	void CreateWindowEvents();

	//* Initialize inputs in this window
	void InitializeInputs();

	//* Make context for this window
	void MakeContext();
    
    //* Creates the main Rendering Window
    bool Create();
    
public:
    
    ContextWindow();
    
    //* Get the width in pixels of the window
    int GetWidth();
    
    //* Get the height in pixels of the window
    int GetHeight();

	//* Resize the size of the window
	void Resize(const unsigned int& Width, const unsigned int& Height);
    
    //* Rename the window title
    void SetWindowName(const WString & NewName);
    
    //* Get the window title name
    WString GetName();
    
    //* Get the aspect of width divided by height in pixels of the window
    float AspectRatio();
    
    //* Creates a Window with a Name, Width and Height
    bool Create(const char * Name, const WindowMode& Mode, const unsigned int& Width, const unsigned int& Height);
    
    //* Asks if window should be closed
    bool ShouldClose();
    
    //* Returns true if window has been created
    bool IsCreated();
    
    //* Total frames drawed since the creation of this window
    unsigned long long GetFrameCount();
    
    //* Get mouse position in screen coordinates relative to the upper left position of this window
    struct Vector2 GetMousePosition(bool Clamp = false);
    
    //* Get key pressed
    bool GetKeyDown(unsigned int Key);
    
    //* Window clear events
    void ClearWindow();
    
    //* Window update frame
    void EndOfFrame();
    
	//* Sets the window icon
	void SetIcon(class Bitmap<UCharRGBA> * Icon);

    //* Window update events
    void PollEvents();
    
    //* Terminates this window
    void Terminate();
    
    //* Set on resized event
    void SetOnResizedEvent(void(*OnWindowResizedFunc)(int Width, int Height));
};
