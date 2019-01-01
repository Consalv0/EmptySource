#pragma once

class CoreApplication {
public:
	static struct ApplicationWindow* MainWindow;

private:
	static bool bInitialized;
	static unsigned long RenderTimeSum;
	
	//* Error Callback related to GLFW
	static void GLFWError(int ErrorID, const char* ErrorDescription);
	
	//* Initialize GLFW Functions using OpenGL Versions, returns true if initialized correctly
	static bool InitializeGLFW(unsigned int VersionMajor, unsigned int VersionMinor);
	
	//* Error Callback related to OpenGL
	static void APIENTRY OGLError(GLenum ErrorSource, GLenum ErrorType, GLuint ErrorID, GLenum ErrorSeverity, GLsizei ErrorLength, const GLchar* ErrorMessage, const void* UserParam);
	
	//* Initialize GLAD OpenGL Functions
	static bool InitalizeGLAD();

	//* Creates the main window for rendering
	static bool InitializeWindow();
	
public:

	//* Prints the Graphics Version info of the videocard used and the GL used
	static void PrintGraphicsInformation();

	//* Initialize the application, it creates a window, a context and loads the OpenGL functions. Returns the error
	static void Initalize();
	
	//* Application Loop, draw is done here
	static void MainLoop();

	//* Terminates window, 
	static void Close();
};