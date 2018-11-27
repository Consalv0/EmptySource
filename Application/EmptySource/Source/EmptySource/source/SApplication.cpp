#include "EmptySource/include/EmptyHeaders.h"
#include "..\include\SWindow.h"
#include "..\include\SApplication.h"

#include "..\include\SMath.h"
#include "..\include\SMesh.h"

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

static void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "GL_";

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "ShaderCompiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "ThirdParty"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Other"; break;
	} std::cout << "<";

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Deprecated"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Undefined"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "PushGroup"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "PopGroup"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Other"; break;
	} std::cout << "><";

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Critical"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Moderate"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Mild"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Notification"; break;
	} std::cout << "><" << id << "> " << message << std::endl;
}

int SApplication::Initalize() {
	if (MainWindow != NULL) return 0;

	if (!glfwInit()) {
		printf("Error :: Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	
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

	glEnable(GL_DEBUG_OUTPUT);
	// glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(glDebugOutput, nullptr);
	// Enable all messages, all sources, all levels, and all IDs:
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

	return 1;
}

void SApplication::MainLoop() {
	///// Temporal Section DELETE AFTER //////

	// Vertex Array Object 
	GLuint TemporalVAO;
	glGenVertexArrays(1, &TemporalVAO);
	glBindVertexArray(TemporalVAO);
	// glDeleteVertexArrays(1, &TemporalVAO);

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
		-0.5F,  0.5F, -0.5F, // 7
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

	SMesh TemporalMesh;
	
	static const SArray<IVector3> TemporalTriangles {
		// Front Face
		IVector3(0, 1, 5),
		IVector3(5, 2, 0),
		// Back Face
		IVector3(4, 3, 7),
		IVector3(7, 6, 4),
		// Right Face
		IVector3(0, 2, 3),
		IVector3(3, 7, 0),
		// Left Face
		IVector3(4, 6, 1),
		IVector3(1, 5, 4),
		// Up Face
		IVector3(4, 3, 2),
		IVector3(2, 5, 4),
		// Down Face
		IVector3(0, 1, 6),
		IVector3(6, 7, 0),
	};
	static const SArray<FVector3> TemporalVertices {
		FVector3( 0.5F, -0.5F, -0.5F), // 0
		FVector3(-0.5F, -0.5F, -0.5F), // 1
		FVector3( 0.5F,  0.5F, -0.5F), // 2
		FVector3( 0.5F,  0.5F,  0.5F), // 3
		FVector3(-0.5F,  0.5F,  0.5F), // 4
		FVector3(-0.5F,  0.5F, -0.5F), // 5
		FVector3(-0.5F, -0.5F,  0.5F), // 6
		FVector3( 0.5F, -0.5F,  0.5F), // 7
	};
	static const SArray<FVector3> TemporalNormals {
		// Front Face
		FVector3( 0.F,  0.F, -1.F), // 0
		// Back Face
		FVector3( 0.F,  0.F,  1.F), // 1
		// Right Face
		FVector3( 1.F,  0.F,  0.F), // 2
		// Left Face
		FVector3(-1.F,  0.F,  0.F), // 3
		// Up Face
		FVector3( 0.F,  1.F,  0.F), // 4
		// Down Face
		FVector3( 0.F, -1.F,  0.F), // 5
	};
	static const SArray<FVector2> TemporalTextureCoords {
		FVector2( 1.0F, -1.0F), // 0
		FVector2(-1.0F, -1.0F), // 1
		FVector2( 1.0F,  1.0F), // 2
		FVector2(-1.0F,  1.0F), // 3
		FVector2(-1.0F,  1.0F), // 4
		FVector2(-1.0F,  1.0F), // 5
		FVector2(-1.0F, -1.0F), // 6
		FVector2( 1.0F, -1.0F), // 7
	};

	TemporalMesh = SMesh(TemporalTriangles, TemporalVertices, TemporalNormals, TemporalTextureCoords, SArray<FVector4>());

	// Generate a Element Buffer for the indices
	GLuint ElementBuffer;
	glGenBuffers(1, &ElementBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElementBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, TemporalMesh.Triangles.size() * sizeof(IVector3), &TemporalMesh.Triangles[0], GL_STATIC_DRAW);

	///////// Give Vertices to OpenGL (This must be done once) //////////////
	// This will identify our vertex buffer
	GLuint VertexBuffer;
	// Generate 1 buffer, put the resulting identifier in VertexBuffer
	glGenBuffers(1, &VertexBuffer);
	// The following commands will talk about our 'VertexBuffer'
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, TemporalMesh.Vertices.size() * sizeof(FVector3), &TemporalMesh.Vertices[0], GL_STATIC_DRAW);

	///////// Give Normals to OpenGL //////////////
	// This will identify our normal buffer
	GLuint NormalBuffer;
	// Generate 1 buffer, put the resulting identifier in NormalBuffer
	glGenBuffers(1, &NormalBuffer);
	// The following commands will talk about our 'NormalBuffer' buffer
	glBindBuffer(GL_ARRAY_BUFFER, NormalBuffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, TemporalMesh.Normals.size() * sizeof(FVector3), &TemporalMesh.Normals[0], GL_STATIC_DRAW);

	///////// Give Texture Coords to OpenGL //////////////
	// This will identify our uv's buffer
	GLuint TextureCoordsBuffer;
	glGenBuffers(1, &TextureCoordsBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, TextureCoordsBuffer);
	glBufferData(GL_ARRAY_BUFFER, TemporalMesh.TextureCoords.size() * sizeof(FVector2), &TemporalMesh.TextureCoords[0], GL_STATIC_DRAW);

	/////////// Creating MVP (ModelMatrix, ViewMatrix, Poryection) Matrix //////////////
	// Perpective matrix (ProjectionMatrix)
	FMatrix4x4 ProjectionMatrix = FMatrix4x4::Perspective(
		45.0F * 0.015708F,			// Aperute angle
		MainWindow->AspectRatio(),	// Aspect ratio
		0.1F,						// Near plane
		200.0F						// Far plane
	);

	FVector3 EyePosition = FVector3(2, 4, 4);

	// Camera rotation, position Matrix
	FMatrix4x4 ViewMatrix = FMatrix4x4::LookAt(
		EyePosition,        // Camera position
		FVector3(0, 0, 0),	// Look position
		FVector3(0, 1, 0)	// Up vector
	);

	// ModelMatrix matrix
	FMatrix4x4 ModelMatrix = FMatrix4x4::Identity();

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
	GLuint         WorldNormalID = glGetUniformLocation(ProgramID, "_WorldNormalMatrix");

	//////////////////////////////////////////

	// Activate Z-buffer
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if its closer to the camera
	glDepthFunc(GL_LESS);
	// Draw Mode
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


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

		double x, y;
		glfwGetCursorPos(MainWindow->Window, &x, &y);
		EyePosition = FVector3(sinf(float(x) * 0.01F) * 2, cosf(float(y) * 0.01F) * 4, 4);

		// Camera rotation, position Matrix
		ViewMatrix = FMatrix4x4::LookAt(
			EyePosition,        // Camera position
			FVector3(0, 0, 0),	// Look position
			FVector3(0, 1, 0)	// Up vector
		);

		glUniformMatrix4fv( ProjectionMatrixID, 1, GL_FALSE,                    ProjectionMatrix.PointerToValue() );
		glUniformMatrix4fv(       ViewMatrixID, 1, GL_FALSE,                          ViewMatrix.PointerToValue() );
		glUniformMatrix4fv(      ModelMatrixID, 1, GL_FALSE,                         ModelMatrix.PointerToValue() ); 
		glUniformMatrix4fv(      WorldNormalID, 1, GL_FALSE, ModelMatrix.Inversed().Transposed().PointerToValue() );
		
		glBindVertexArray(TemporalVAO);

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

		// 3st attribute buffer : textureCoords
		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, TextureCoordsBuffer);
		glVertexAttribPointer(
			2,                  // attribute 1
			2,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// Index buffer
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElementBuffer);
		
		// Draw the triangles !
		glDrawElements(
			GL_TRIANGLES,	 // mode
			12 * 3,			 // count
			GL_UNSIGNED_INT, // type
			(void*)0		 // element array buffer offset
		);
		// glDrawArrays(GL_TRIANGLES, 0, sizeof(TemporalMesh.Vertices) * sizeof(FVector3)); // Starting from vertex 0; to vertices total
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

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