#pragma once

class CoreApplication {
public:
	static struct ApplicationWindow* MainWindow;

private:
	static bool bInitialized;
	static double RenderTimeSum;
	
	//* Initialize GLFW Functions using OpenGL Versions, returns true if initialized correctly
	static bool InitializeGLFW(unsigned int VersionMajor, unsigned int VersionMinor);
	
	//* Initialize GLAD OpenGL Functions
	static bool InitalizeGLAD();

	//* Creates the main window for rendering
	static bool InitializeWindow();

	//* Initialize Nvidia Managment Library or IOKitLib
	static bool InitializeDeviceFunctions();
    
public:

	//* Initialize the application, it creates a window, a context and loads the OpenGL functions. Returns the error
	static void Initalize();
	
	//* Application Loop, draw is done here
	static void MainLoop();

	//* Terminates window, 
	static void Close();
};
