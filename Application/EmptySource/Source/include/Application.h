#pragma once

class CoreApplication {
private:
	static bool bInitialized;
	static double RenderTimeSum;
	
	//* Initialize SDL Functions using OpenGL Versions, returns true if initialized correctly
	static bool InitializeSDL(unsigned int VersionMajor, unsigned int VersionMinor);
	
	//* Initialize GLAD OpenGL Functions
	static bool InitalizeGLAD();

	//* Creates the main window for rendering
	static bool InitializeWindow();
    
public:
	static struct ContextWindow & GetMainWindow();

	//* Initialize the application, it creates a window, a context and loads the OpenGL functions. Returns the error
	static void Initalize();
	
	//* Application Loop, draw is done here
	static void MainLoop();

	//* Terminates Application
	static void Terminate();
};
