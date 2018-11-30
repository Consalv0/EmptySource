#include "EmptySource/include/EmptyHeaders.h"
#include "..\include\SWindow.h"
#include "..\include\SApplication.h"

#include "..\include\SMath.h"
#include "..\include\SMesh.h"

SApplication::SApplication() {
	MainWindow = NULL;
	bInitialized = false;
}

void SApplication::GLFWError(int id, const char* desc) {
	fprintf(stderr, desc);
}

void APIENTRY SApplication::OGLError(GLenum ErrorSource, GLenum ErrorType, GLuint ErrorID, GLenum ErrorSeverity, GLsizei ErrorLength, const GLchar * ErrorMessage, const void * UserParam)
{
	// ignore non-significant error/warning codes
	if (ErrorID == 131169 || ErrorID == 131185 || ErrorID == 131218 || ErrorID == 131204) return;

	std::cout << "GL_";

	switch (ErrorSource)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "ShaderCompiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "ThirdParty"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Other"; break;
	} std::cout << "<";

	switch (ErrorType)
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

	switch (ErrorSeverity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Critical"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Moderate"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Mild"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Notification"; break;
	} std::cout << "><" << ErrorID << "> " << ErrorMessage << std::endl;
}

bool SApplication::InitalizeGLAD() {
	if (!gladLoadGL()) {
		printf("Error :: Unable to load OpenGL functions!\n");
		return false;
	}

	glEnable(GL_DEBUG_OUTPUT);
	// glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(OGLError, nullptr);
	// Enable all messages, all sources, all levels, and all IDs:
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

	return true;
}

void SApplication::GraphicsInformation() {
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

bool SApplication::InitializeGLFW(unsigned int VersionMajor, unsigned int VersionMinor) {
	if (!glfwInit()) {
		printf("Error :: Failed to initialize GLFW\n");
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, VersionMajor);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, VersionMinor);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

	glfwSetErrorCallback(&SApplication::GLFWError);
	return true;
}

void SApplication::Initalize() {
	if (bInitialized) return;
	if (InitializeGLFW(4, 6) == false) return; 

	MainWindow = new SWindow();

	if (MainWindow->Create("EmptySource - Debug", ES_WINDOW_MODE_WINDOWED, 1366, 768)) {
		printf("Error :: Application Window couldn't be created!\n");
		glfwTerminate();
		return;
	}

	MainWindow->MakeContext();
	MainWindow->InitializeInputs();

	if (InitalizeGLAD() == false) return;

	bInitialized = true;
}

void SApplication::MainLoop() {
	///// Temporal Section DELETE AFTER //////

	// Vertex Array Object 
	GLuint TemporalVAO;
	glGenVertexArrays(1, &TemporalVAO);
	glBindVertexArray(TemporalVAO);
	// glDeleteVertexArrays(1, &TemporalVAO);

	/*
	* If the first vertex is (-1, -1, 0). This means that unless we transform it in some way,
	* it will be displayed at (-1, -1) on the screen. What does this mean? The screen origin is in the middle,
	* X is on the right, as usual, and Y is up. 
	*/
	SMesh TemporalMesh;
	static const SMeshTriangles TemporalTriangles {
		// Front Face
		{  0,  1,  2 }, {  2,  3,  0 },
		// Back Face
		{  4,  5,  6 }, {  6,  7,  4 },
		// Right Face
		{  8,  9, 10 }, { 10, 11,  8 },
		// Left Face
		{ 12, 13, 14 }, { 14, 15, 12 },
		// Up Face
		{ 16, 17, 18 }, { 18, 19, 16 },
		// Down Face
		{ 20, 21, 22 }, { 22, 23, 20 },
	};
	static const SMeshVector3D TemporalVertices {
		// Front Face
		{ 0.5F, -0.5F, -0.5F }, // 1 : 1
		{-0.5F, -0.5F, -0.5F }, // 2 : 2
		{-0.5F,  0.5F, -0.5F }, // 6 : 3
		{ 0.5F,  0.5F, -0.5F }, // 3 : 5

		// Back Face
		{ -0.5F,  0.5F,  0.5F }, // 5 : 7
		{  0.5F,  0.5F,  0.5F }, // 4 : 8
		{  0.5F, -0.5F,  0.5F }, // 8 : 10
		{ -0.5F, -0.5F,  0.5F }, // 7 : 11

		// Right Face
		{ 0.5F, -0.5F, -0.5F }, // 1 : 13
		{ 0.5F,  0.5F, -0.5F }, // 3 : 14
		{ 0.5F,  0.5F,  0.5F }, // 4 : 16
		{ 0.5F, -0.5F,  0.5F }, // 8 : 17

		// Left Face
		{-0.5F,  0.5F,  0.5F }, // 5 : 19
		{-0.5F, -0.5F,  0.5F }, // 7 : 20
		{-0.5F, -0.5F, -0.5F }, // 2 : 22
		{-0.5F,  0.5F, -0.5F }, // 6 : 23

		// Up Face
		{-0.5F,  0.5F,  0.5F }, // 5 : 25
		{ 0.5F,  0.5F,  0.5F }, // 4 : 26
		{ 0.5F,  0.5F, -0.5F }, // 3 : 28
		{-0.5F,  0.5F, -0.5F }, // 6 : 29

		// Down Face
		{ 0.5F, -0.5F, -0.5F }, // 1 : 31
		{-0.5F, -0.5F, -0.5F }, // 2 : 32
		{-0.5F, -0.5F,  0.5F }, // 7 : 34
		{ 0.5F, -0.5F,  0.5F }, // 8 : 35
	};
	static const SMeshVector3D TemporalNormals {
		// Front Face
		{ 0.0F,  0.0F, -1.0F }, // 1
		{ 0.0F,  0.0F, -1.0F }, // 2
		{ 0.0F,  0.0F, -1.0F }, // 6
		{ 0.0F,  0.0F, -1.0F }, // 3

		// Back Face		 
		{ 0.0F,  0.0F,  1.0F }, // 5
		{ 0.0F,  0.0F,  1.0F }, // 4
		{ 0.0F,  0.0F,  1.0F }, // 8
		{ 0.0F,  0.0F,  1.0F }, // 7

		// Right Face		 
		{ 1.0F,  0.0F,  0.0F }, // 1
		{ 1.0F,  0.0F,  0.0F }, // 3
		{ 1.0F,  0.0F,  0.0F }, // 4
		{ 1.0F,  0.0F,  0.0F }, // 8

		// Left Face		 
		{-1.0F,  0.0F,  0.0F }, // 5
		{-1.0F,  0.0F,  0.0F }, // 7
		{-1.0F,  0.0F,  0.0F }, // 2
		{-1.0F,  0.0F,  0.0F }, // 6

		// Up Face			 
		{ 0.0F,  1.0F,  0.0F }, // 5
		{ 0.0F,  1.0F,  0.0F }, // 4
		{ 0.0F,  1.0F,  0.0F }, // 3
		{ 0.0F,  1.0F,  0.0F }, // 6

		// Down Face		 
		{ 0.0F, -1.0F,  0.0F }, // 2
		{ 0.0F, -1.0F,  0.0F }, // 1
		{ 0.0F, -1.0F,  0.0F }, // 7
		{ 0.0F, -1.0F,  0.0F }, // 8
	};
	static const SMeshUVs      TemporalTextureCoords {
		// Front Face
		{ 1.0F, -1.0F }, // 1
		{-1.0F, -1.0F }, // 2
		{-1.0F,  1.0F }, // 6
		{ 1.0F,  1.0F }, // 3

		// Back Face
		{-1.0F,  1.0F }, // 5
		{ 1.0F,  1.0F }, // 4
		{ 1.0F, -1.0F }, // 8
		{-1.0F, -1.0F }, // 7

		// Right Face
		{ 1.0F, -1.0F }, // 1
		{ 1.0F,  1.0F }, // 3
		{ 1.0F,  1.0F }, // 4
		{ 1.0F, -1.0F }, // 8

		// Left Face
		{-1.0F,  1.0F }, // 5
		{-1.0F, -1.0F }, // 7
		{-1.0F, -1.0F }, // 2
		{-1.0F,  1.0F }, // 6
		
		// Up Face
		{-1.0F,  1.0F }, // 5
		{ 1.0F,  1.0F }, // 4
		{ 1.0F,  1.0F }, // 3
		{-1.0F,  1.0F }, // 6
		
		// Down Face
		{ 1.0F, -1.0F }, // 1
		{-1.0F, -1.0F }, // 2
		{-1.0F, -1.0F }, // 7
		{ 1.0F, -1.0F }, // 8
	};
	static const SMeshColors   TemporalColors {
		// Front Face
		{ 0.0F, 0.0F, 1.0F, 1.0F },
		{ 0.0F, 0.0F, 1.0F, 1.0F },
		{ 0.0F, 0.0F, 1.0F, 1.0F },
		{ 0.0F, 0.0F, 1.0F, 1.0F },

		// Back Face
		{ 0.0F, 1.0F, 0.0F, 1.0F },
		{ 0.0F, 1.0F, 0.0F, 1.0F },
		{ 0.0F, 1.0F, 0.0F, 1.0F },
		{ 0.0F, 1.0F, 0.0F, 1.0F },

		// Right Face
		{ 1.0F, 0.0F, 0.0F, 1.0F },
		{ 1.0F, 0.0F, 0.0F, 1.0F },
		{ 1.0F, 0.0F, 0.0F, 1.0F },
		{ 1.0F, 0.0F, 0.0F, 1.0F },

		// Left Face
		{ 0.0F, 1.0F, 1.0F, 1.0F },
		{ 0.0F, 1.0F, 1.0F, 1.0F },
		{ 0.0F, 1.0F, 1.0F, 1.0F },
		{ 0.0F, 1.0F, 1.0F, 1.0F },

		// Up Face
		{ 1.0F, 1.0F, 0.0F, 1.0F },
		{ 1.0F, 1.0F, 0.0F, 1.0F },
		{ 1.0F, 1.0F, 0.0F, 1.0F },
		{ 1.0F, 1.0F, 0.0F, 1.0F },

		// Down Face
		{ 1.0F, 1.0F, 1.0F, 1.0F },
		{ 1.0F, 1.0F, 1.0F, 1.0F },
		{ 1.0F, 1.0F, 1.0F, 1.0F },
		{ 1.0F, 1.0F, 1.0F, 1.0F },
	};

	TemporalMesh = SMesh(TemporalTriangles, TemporalVertices, TemporalNormals, TemporalTextureCoords, TemporalColors);

	///////// Give Vertices to OpenGL (This must be done once) //////////////
	// This will identify our vertex buffer
	GLuint VertexBuffer;
	// Generate 1 buffer, put the resulting identifier in VertexBuffer
	glGenBuffers(1, &VertexBuffer);
	// The following commands will talk about our 'VertexBuffer'
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, TemporalMesh.Vertices.size() * sizeof(Vertex), &TemporalMesh.Vertices[0], GL_STATIC_DRAW);

	// Generate a Element Buffer for the indices
	GLuint ElementBuffer;
	glGenBuffers(1, &ElementBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElementBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, TemporalMesh.Triangles.size() * sizeof(IVector3), &TemporalMesh.Triangles[0], GL_STATIC_DRAW);

	// set the vertex attribute pointers
	// vertex Positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	// vertex normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
	// vertex tangent
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
	// vertex texture coords
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, UV0));
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, UV1));
	// vertex color
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Color));

	SArray<FMatrix4x4> Matrices;

	Matrices.push_back(FMatrix4x4());
	srand((unsigned int)glfwGetTime());
	for (size_t i = 0; i < 100000; i++) {
		Matrices.push_back(FMatrix4x4::Translate({ ((rand() % 2000) * 0.5F) - 500, ((rand() % 2000) * 0.5F) - 500, ((rand() % 2000) * 0.5F) - 500 }));
	}

	///////// Give Matrices to OpenGL //////////////
	GLuint ModelMatrixBuffer;
	glGenBuffers(1, &ModelMatrixBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
	glBufferData(GL_ARRAY_BUFFER, Matrices.size() * sizeof(FMatrix4x4), &Matrices[0], GL_STATIC_DRAW);

	// set transformation matrices as an instance vertex attribute (with divisor 1)
	// note: we're cheating a little by taking the, now publicly declared, VAO of the model's mesh(es) and adding new vertexAttribPointers
	// normally you'd want to do this in a more organized fashion, but for learning purposes this will do.
	// -----------------------------------------------------------------------------------------------------------------------------------
	// set attribute pointers for matrix (4 times vec4)
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(FMatrix4x4), (void*)0);
	glEnableVertexAttribArray(7);
	glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(FMatrix4x4), (void*)(sizeof(FVector4)));
	glEnableVertexAttribArray(8);
	glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, sizeof(FMatrix4x4), (void*)(2 * sizeof(FVector4)));
	glEnableVertexAttribArray(9);
	glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, sizeof(FMatrix4x4), (void*)(3 * sizeof(FVector4)));

	glVertexAttribDivisor(6, 1);
	glVertexAttribDivisor(7, 1);
	glVertexAttribDivisor(8, 1);
	glVertexAttribDivisor(9, 1);

	glBindVertexArray(0);

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
	// GLuint         ModelMatrixID = glGetUniformLocation(ProgramID, "_ModelMatrix");
	// GLuint         WorldNormalID = glGetUniformLocation(ProgramID, "_WorldNormalMatrix");

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

		FVector2 CursorPosition = MainWindow->GetMousePosition();
		EyePosition = FVector3(sinf(float(CursorPosition.x) * 0.01F) * 2, cosf(float(CursorPosition.y) * 0.01F) * 4, cosf(float(CursorPosition.y) * 0.01F) * 4);

		// Camera rotation, position Matrix
		ViewMatrix = FMatrix4x4::LookAt(
			EyePosition,        // Camera position
			FVector3(0, 0, 0),	// Look position
			FVector3(0, 1, 0)	// Up vector
		);

		glUniformMatrix4fv( ProjectionMatrixID, 1, GL_FALSE,                    ProjectionMatrix.PointerToValue() );
		glUniformMatrix4fv(       ViewMatrixID, 1, GL_FALSE,                          ViewMatrix.PointerToValue() );
		// glUniformMatrix4fv(      ModelMatrixID, 1, GL_FALSE,                         ModelMatrix.PointerToValue() ); 
		// glUniformMatrix4fv(      WorldNormalID, 1, GL_FALSE, ModelMatrix.Inversed().Transposed().PointerToValue() );
		
		glBindVertexArray(TemporalVAO);
		
		// Draw the triangles !
		glDrawElementsInstanced (
			GL_TRIANGLES,	                        // mode
			(int)TemporalMesh.Triangles.size() * 3,	// count
			GL_UNSIGNED_INT,                        // type
			(void*)0,		                        // element array buffer offset
			(GLsizei)Matrices.size()                                       // element count
		);
		// glDrawArrays(GL_TRIANGLES, 0, sizeof(TemporalMesh.Vertices) * sizeof(FVector3)); // Starting from vertex 0; to vertices total
		
		glBindVertexArray(0);

		MainWindow->EndOfFrame();

	} while (
		MainWindow->ShouldClose() == false && MainWindow->GetKeyPressed(GLFW_KEY_ESCAPE)
	);
}

void SApplication::Close() {
	MainWindow->Terminate();
	glfwTerminate();
};