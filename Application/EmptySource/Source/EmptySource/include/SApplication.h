#pragma once

class SApplication {
public:
	struct SWindow* MainWindow;
private:

public:
	SApplication();
	
	static void glfwPrintError(int ID, const char* Description);
	
	//* Prints the Graphics Version info of the videocard used and the GL used
	static void GetGraphicsVersionInformation();

	//* Initialize the application, it creates a window, a context and loads the OpenGL functions. Returns the error
	int Initalize();
	
	//* Application Loop, draw is done here
	void MainLoop();

	void Close();
};