#include "EmptySource/include/EmptyHeaders.h"
#include "..\include\SWindow.h"
#include "..\include\SApplication.h"

#include "..\include\SMath.h"

SApplication::SApplication() {
	MainWindow = NULL;
}

void SApplication::glfwPrintError(int id, const char* desc) {
	fprintf(stderr, desc);
}

void SApplication::GetGraphicsVersionInformation() {
	const GLubyte    *renderer = glGetString(GL_RENDERER);
	const GLubyte      *vendor = glGetString(GL_VENDOR);
	const GLubyte     *version = glGetString(GL_VERSION);
	const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	GLint major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	printf("GC Vendor            : %s\n", vendor);
	printf("GC Renderer          : %s\n", renderer);
	printf("GL Version (string)  : %s\n", version);
	printf("GL Version (integer) : %d.%d\n", major, minor);
	printf("GLSL Version         : %s\n", glslVersion);
}

int SApplication::Initalize() {
	if (MainWindow != NULL) return 0;
	
	glfwSetErrorCallback(&SApplication::glfwPrintError);
	printf("Initalizing Application:\n");

	MainWindow = new SWindow();

	if (MainWindow->Create("EmptySource - Debug", ES_WINDOW_MODE_WINDOWED, 1366, 768) || MainWindow->Window == NULL) {
		printf("Error :: Application Window couldn't be created!\n");
		printf("\nPress any key to close...\n");
		glfwTerminate();
		_getch();
		return -1;
	}

	MainWindow->MakeContext();
	MainWindow->InitializeInputs();

	if (!gladLoadGL()) {
		printf("Error :: Unable to load OpenGL functions!\n");
		return -1;
	}

	return 1;
}

void SApplication::MainLoop() {
	///// Temporal Section DELETE AFTER //////

	// Vertex Array Object 
	GLuint TemporalVAO;
	glGenVertexArrays(1, &TemporalVAO);
	glBindVertexArray(TemporalVAO);

	// Activate Z-buffer
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if its closer to the camera
	glDepthFunc(GL_LESS);
	// Draw Mode
	glPolygonMode(GL_FRONT, GL_FILL);

	/*
	* An array of 3 vectors which represents 3 vertices
	* The first vertex is (-1, -1, 0). This means that unless we transform it in some way,
	* it will be displayed at (-1, -1) on the screen. What does this mean? The screen origin is in the middle,
	* X is on the right, as usual, and Y is up. 
	*/
	static const GLfloat TemporalVertexBufferScene[] = {
		// Front Face
		 0.5F, -0.5F, -0.5F, // 1
		-0.5F, -0.5F, -0.5F, // 2
		-0.5F,  0.5F, -0.5F, // 6
		-0.5F,  0.5F, -0.5F, // 6
		 0.5F,  0.5F, -0.5F, // 3
		 0.5F, -0.5F, -0.5F, // 1

		// Back Face
		-0.5F,  0.5F,  0.5F, // 5
		 0.5F,  0.5F,  0.5F, // 4
		 0.5F, -0.5F,  0.5F, // 8
		 0.5F, -0.5F,  0.5F, // 8
		-0.5F, -0.5F,  0.5F, // 7
		-0.5F,  0.5F,  0.5F, // 5

		// Right Face
		 0.5F, -0.5F, -0.5F, // 1
		 0.5F,  0.5F, -0.5F, // 3
		 0.5F,  0.5F,  0.5F, // 4
		 0.5F,  0.5F,  0.5F, // 4
		 0.5F, -0.5F,  0.5F, // 8
		 0.5F, -0.5F, -0.5F, // 1

		// Left Face
		-0.5F,  0.5F,  0.5F, // 5
		-0.5F, -0.5F,  0.5F, // 7
		-0.5F, -0.5F, -0.5F, // 2
		-0.5F, -0.5F, -0.5F, // 2
		-0.5F,  0.5F, -0.5F, // 6
		-0.5F,  0.5F,  0.5F, // 5

		// Up Face
		-0.5F,  0.5F,  0.5F, // 5
		 0.5F,  0.5F,  0.5F, // 4
		 0.5F,  0.5F, -0.5F, // 3
		 0.5F,  0.5F, -0.5F, // 3
		-0.5F,  0.5F, -0.5F, // 6
		-0.5F,  0.5F,  0.5F, // 5

		// Down Face
		 0.5F, -0.5F, -0.5F, // 1
		-0.5F, -0.5F, -0.5F, // 2
		-0.5F, -0.5F,  0.5F, // 7
		-0.5F, -0.5F,  0.5F, // 7
		 0.5F, -0.5F,  0.5F, // 8
		 0.5F, -0.5F, -0.5F, // 1
	};

	static const GLfloat TemporalTextureCoordsBufferScene[] = {
		// Front Face
		 1.0F, -1.0F, -1.0F, // 1
		-1.0F, -1.0F, -1.0F, // 2
		-1.0F,  1.0F, -1.0F, // 6
		-1.0F,  1.0F, -1.0F, // 6
		 1.0F,  1.0F, -1.0F, // 3
		 1.0F, -1.0F, -1.0F, // 1
							 
		// Back Face		 
		-1.0F,  1.0F,  1.0F, // 5
		 1.0F,  1.0F,  1.0F, // 4
		 1.0F, -1.0F,  1.0F, // 8
		 1.0F, -1.0F,  1.0F, // 8
		-1.0F, -1.0F,  1.0F, // 7
		-1.0F,  1.0F,  1.0F, // 5
							 
		// Right Face		 
		 1.0F, -1.0F, -1.0F, // 1
		 1.0F,  1.0F, -1.0F, // 3
		 1.0F,  1.0F,  1.0F, // 4
		 1.0F,  1.0F,  1.0F, // 4
		 1.0F, -1.0F,  1.0F, // 8
		 1.0F, -1.0F, -1.0F, // 1
							 
		// Left Face		 
		-1.0F,  1.0F,  1.0F, // 5
		-1.0F, -1.0F,  1.0F, // 7
		-1.0F, -1.0F, -1.0F, // 2
		-1.0F, -1.0F, -1.0F, // 2
		-1.0F,  1.0F, -1.0F, // 6
		-1.0F,  1.0F,  1.0F, // 5

		// Up Face
		-1.0F,  1.0F,  1.0F, // 5
		 1.0F,  1.0F,  1.0F, // 4
		 1.0F,  1.0F, -1.0F, // 3
		 1.0F,  1.0F, -1.0F, // 3
		-1.0F,  1.0F, -1.0F, // 6
		-1.0F,  1.0F,  1.0F, // 5

		// Down Face
		 1.0F, -1.0F, -1.0F, // 1
		-1.0F, -1.0F, -1.0F, // 2
		-1.0F, -1.0F,  1.0F, // 7
		-1.0F, -1.0F,  1.0F, // 7
		 1.0F, -1.0F,  1.0F, // 8
		 1.0F, -1.0F, -1.0F, // 1
	};

	static const GLfloat TemporalNormalsBufferScene[] = {
		// Front Face
		 0.0F,  0.0F, -1.0F, // 1
		 0.0F,  0.0F, -1.0F, // 2
		 0.0F,  0.0F, -1.0F, // 6
		 0.0F,  0.0F, -1.0F, // 6
		 0.0F,  0.0F, -1.0F, // 3
		 0.0F,  0.0F, -1.0F, // 1
							 
		// Back Face		 
		 0.0F,  0.0F,  1.0F, // 5
		 0.0F,  0.0F,  1.0F, // 4
		 0.0F,  0.0F,  1.0F, // 8
		 0.0F,  0.0F,  1.0F, // 8
		 0.0F,  0.0F,  1.0F, // 7
		 0.0F,  0.0F,  1.0F, // 5
							 
		// Right Face		 
		 1.0F,  0.0F,  0.0F, // 1
		 1.0F,  0.0F,  0.0F, // 3
		 1.0F,  0.0F,  0.0F, // 4
		 1.0F,  0.0F,  0.0F, // 4
		 1.0F,  0.0F,  0.0F, // 8
		 1.0F,  0.0F,  0.0F, // 1
							 
		// Left Face		 
		-1.0F,  0.0F,  0.0F, // 5
		-1.0F,  0.0F,  0.0F, // 7
		-1.0F,  0.0F,  0.0F, // 2
		-1.0F,  0.0F,  0.0F, // 2
		-1.0F,  0.0F,  0.0F, // 6
		-1.0F,  0.0F,  0.0F, // 5
							 
		// Up Face			 
		 0.0F,  1.0F,  0.0F, // 5
		 0.0F,  1.0F,  0.0F, // 4
		 0.0F,  1.0F,  0.0F, // 3
		 0.0F,  1.0F,  0.0F, // 3
		 0.0F,  1.0F,  0.0F, // 6
		 0.0F,  1.0F,  0.0F, // 5
							 
		// Down Face		 
		 0.0F, -1.0F,  0.0F, // 1
		 0.0F, -1.0F,  0.0F, // 2
		 0.0F, -1.0F,  0.0F, // 7
		 0.0F, -1.0F,  0.0F, // 7
		 0.0F, -1.0F,  0.0F, // 8
		 0.0F, -1.0F,  0.0F, // 1
	};

	///////// Give Vertices to OpenGL (This must be done once) //////////////
	// This will identify our vertex buffer
	GLuint VertexBuffer;
	// Generate 1 buffer, put the resulting identifier in VertexBuffer
	glGenBuffers(1, &VertexBuffer);
	// The following commands will talk about our 'VertexBuffer'
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, sizeof(TemporalVertexBufferScene), TemporalVertexBufferScene, GL_STATIC_DRAW);

	///////// Give Normals to OpenGL //////////////
	// This will identify our normal buffer
	GLuint NormalBuffer;
	// Generate 1 buffer, put the resulting identifier in NormalBuffer
	glGenBuffers(1, &NormalBuffer);
	// The following commands will talk about our 'NormalBuffer' buffer
	glBindBuffer(GL_ARRAY_BUFFER, NormalBuffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, sizeof(TemporalTextureCoordsBufferScene), TemporalTextureCoordsBufferScene, GL_STATIC_DRAW);

	/////////// Creating MVP (ModelMatrix, ViewMatrix, Poryection) Matrix //////////////
	// Perpective matrix (ProjectionMatrix)
	FMatrix4x4 ProjectionMatrix = FMatrix4x4::Perspective(
		45.0F * 0.015708F,			// Aperute angle
		MainWindow->AspectRatio(),	// Aspect ratio
		0.1F,						// Near plane
		200.0F						// Far plane
	);

	FVector3 EyePosition = FVector3(0, 5, -5);

	// Camera rotation, position Matrix
	FMatrix4x4 ViewMatrix = FMatrix4x4::LookAt(
		EyePosition,        // Camera position
		FVector3(0, 0, 0),	// Look position
		FVector3(0, 1, 0)	// Up vector
	);

	// ModelMatrix matrix
	FMatrix4x4 ModelMatrix = FMatrix4x4::Identity();

	FVector4 Holi = FVector4(1, 2, 3, 4);
	FVector2 Holi2 = Holi;
	Holi = Holi2*Holi;

	// MVP matrix
	FMatrix4x4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

	///////// Create and compile our GLSL program from the shaders //////////////
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream("Data\\Shaders\\UnlitBase.vertex.glsl", std::ios::in);
	if (VertexShaderStream.is_open()) {
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		try {
			VertexShaderCode = sstr.str();
		} catch (...) {
			return;
		}
		VertexShaderStream.close();
	} else {
		printf("Impossible to open \"%s\". Are you in the right directory ?\n", "Data\\Shaders\\UnlitBase.vertex.glsl");
		return;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream("Data\\Shaders\\UnlitBase.fragment.glsl", std::ios::in);
	if (FragmentShaderStream.is_open()) {
		std::stringstream sstr;
		try {
			sstr << FragmentShaderStream.rdbuf();
		} catch (...) {
			return;
		}
		FragmentShaderCode = sstr.str();
		FragmentShaderStream.close();
	} else {
		printf("Impossible to open \"%s\". Are you in the right directory ?\n", "Data\\Shaders\\UnlitBase.fragment.glsl");
		return;
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	printf("Compiling shader : %s\n", "Data\\Shaders\\UnlitBase.vertex.glsl");
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);

		glDeleteShader(VertexShaderID);
		glDeleteShader(FragmentShaderID);
		return;
	}

	// Compile Fragment Shader
	printf("Compiling shader : %s\n", "Data\\Shaders\\UnlitBase.fragment.glsl");
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);

		glDeleteShader(VertexShaderID);
		glDeleteShader(FragmentShaderID);
		return;
	}

	// Link the shader program
	printf("Linking shader program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);

		glDeleteShader(VertexShaderID);
		glDeleteShader(FragmentShaderID);
		glDeleteProgram(ProgramID);
		return;
	}
	
	if (ProgramID == GL_FALSE) {
		glDetachShader(ProgramID, VertexShaderID);
		glDetachShader(ProgramID, FragmentShaderID);

		glDeleteShader(VertexShaderID);
		glDeleteShader(FragmentShaderID);
	}

	///////// Give Uniforms to GLSL /////////////
	// Get the ID of the uniforms
	GLuint    ProjectionMatrixID = glGetUniformLocation(ProgramID, "_ProjectionMatrix");
	GLuint          ViewMatrixID = glGetUniformLocation(ProgramID, "_ViewMatrix");
	GLuint         ModelMatrixID = glGetUniformLocation(ProgramID, "_ModelMatrix");

	//////////////////////////////////////////

	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(ProgramID);

		//////// Drawing ModelMatrix ////////
		ProjectionMatrix = FMatrix4x4::Perspective(
			45.0F * 0.015708F,			// Aperute angle
			MainWindow->AspectRatio(),	// Aspect ratio
			0.1F,						// Near plane
			200.0F						// Far plane
		);

		glUniformMatrix4fv( ProjectionMatrixID, 1, GL_FALSE, ProjectionMatrix.PoiterToValue() );
		glUniformMatrix4fv(       ViewMatrixID, 1, GL_FALSE,       ViewMatrix.PoiterToValue() );
		glUniformMatrix4fv(      ModelMatrixID, 1, GL_FALSE,      ModelMatrix.PoiterToValue() );

		// 1st attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// 2st attribute buffer : normals
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, NormalBuffer);
		glVertexAttribPointer(
			1,                  // attribute 1
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// Draw the model !
		glDrawArrays(GL_TRIANGLES, 0, sizeof(TemporalVertexBufferScene)); // Starting from vertex 0; to vertices total
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		glfwSwapBuffers(MainWindow->Window);
		glfwPollEvents();

	} while (
		MainWindow->ShouldClose() == false && 
		glfwGetKey(MainWindow->Window, GLFW_KEY_ESCAPE) != GLFW_PRESS
	);
}

void SApplication::Close() {
	MainWindow->Destroy();
	glfwTerminate();
};