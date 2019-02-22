#include "..\include\Core.h"
#include "..\include\Math\Math.h"

#include "..\include\FileManager.h"
#include "..\include\MeshLoader.h"

#include "..\include\Graphics.h"
#include "..\include\Utility\DeviceTemperature.h"
#include "..\include\Utility\CUDAUtility.h"
#include "..\include\Utility\LogMath.h"
#include "..\include\Time.h"
#include "..\include\Window.h"
#include "..\include\Application.h"
#include "..\include\Mesh.h"
#include "..\include\ShaderProgram.h"
#include "..\include\Object.h"
#include "..\include\Space.h"
#include "..\include\Material.h"
#include "..\include\Utility\Timer.h"
#include "..\include\RenderTarget.h"
#include "..\include\Texture2D.h"
#include "..\include\Texture3D.h"

#include <nvml.h>

bool RunTest(int N, MeshVertex * Positions);
int FindBoundingBox(int N, MeshVertex * Vertices);
int VoxelizeToTexture3D(Texture3D* Texture, int N, MeshVertex * Vertices);
int RayTracingTexture2D(Texture2D* texture);

ApplicationWindow* CoreApplication::MainWindow = NULL;
bool CoreApplication::bInitialized = false;
unsigned long CoreApplication::RenderTimeSum = 0;

bool CoreApplication::InitalizeGLAD() {
	if (!gladLoadGL()) {
		Debug::Log(Debug::LogCritical, L"Unable to load OpenGL functions!");
		return false;
	}

	glEnable(GL_DEBUG_OUTPUT);
	// glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(Debug::OGLError, nullptr);
	// --- Enable all messages, all sources, all levels, and all IDs:
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

	Debug::PrintGraphicsInformation();

	return true;
}

bool CoreApplication::InitializeWindow() {
	MainWindow = new ApplicationWindow();

	if (MainWindow->Create("EmptySource - Debug", WindowMode::Windowed, 1366, 768)) {
		Debug::Log(Debug::LogCritical, L"Application Window couldn't be created!");
		glfwTerminate();
		return false;
	}

	MainWindow->MakeContext();
	MainWindow->InitializeInputs();

	return true;
}

bool CoreApplication::InitializeNVML() {
	nvmlReturn_t DeviceResult = nvmlInit();

	if (NVML_SUCCESS != DeviceResult) {
		Debug::Log(Debug::LogError, L"NVML :: Failed to initialize: %s", CharToWChar(nvmlErrorString(DeviceResult)));
		return false;
	}

	return true;
}

bool CoreApplication::InitializeGLFW(unsigned int VersionMajor, unsigned int VersionMinor) {
	if (!glfwInit()) {
		Debug::Log(Debug::LogCritical, L"Failed to initialize GLFW\n");
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, VersionMajor);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, VersionMinor);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

	glfwSetErrorCallback(&Debug::GLFWError);
	return true;
}

void CoreApplication::Initalize() {
	if (bInitialized) return;
	if (InitializeGLFW(4, 6) == false) return; 
	if (InitializeWindow() == false) return;
	if (InitalizeGLAD() == false) return;
	if (InitializeNVML() == false) return;

	CUDA::FindCudaDevice();

	bInitialized = true;
}

void CoreApplication::Close() {
	MainWindow->Terminate();
	delete MainWindow;

	glfwTerminate(); 
	nvmlShutdown();
};

void CoreApplication::MainLoop() {
	
	Space NewSpace; Space OtherNewSpace(NewSpace);
	Object* GObject = Space::GetFirstSpace()->MakeObject();
	GObject->Delete();

	/////////// Creating MVP (ModelMatrix, ViewMatrix, Poryection) Matrix //////////////
	// --- Perpective matrix (ProjectionMatrix)
	Matrix4x4 ProjectionMatrix;

	Vector3 EyePosition = 0;
	Vector3 LightPosition = Vector3(1, 0);
	// --- Camera rotation, position Matrix
	float ViewSpeed = 5;
	Vector3 ViewOrientation;
	Matrix4x4 ViewMatrix;

	// --- Create and compile our GLSL shader programs from text files
	ShaderStage VertexBase =        ShaderStage(ShaderType::Vertex, FileManager::Open(L"Data\\Shaders\\Base.vertex.glsl"));
	ShaderStage VoxelizerVertex =   ShaderStage(ShaderType::Vertex, FileManager::Open(L"Data\\Shaders\\Voxelizer.vertex.glsl"));
	ShaderStage FragmentBRDF =      ShaderStage(ShaderType::Fragment, FileManager::Open(L"Data\\Shaders\\BRDF.fragment.glsl"));
	ShaderStage PassthroughVertex = ShaderStage(ShaderType::Vertex, FileManager::Open(L"Data\\Shaders\\Passthrough.vertex.glsl"));
	ShaderStage FragRenderTexture = ShaderStage(ShaderType::Fragment, FileManager::Open(L"Data\\Shaders\\RenderTexture.fragment.glsl"));
	ShaderStage FragmentUnlit =     ShaderStage(ShaderType::Fragment, FileManager::Open(L"Data\\Shaders\\Unlit.fragment.glsl"));
	ShaderStage Voxelizer =         ShaderStage(ShaderType::Geometry, FileManager::Open(L"Data\\Shaders\\Voxelizer.geometry.glsl"));

	ShaderProgram VoxelBRDFShader = ShaderProgram(L"VoxelBRDF");
	VoxelBRDFShader.AppendStage(&VoxelizerVertex);
	VoxelBRDFShader.AppendStage(&Voxelizer);
	VoxelBRDFShader.AppendStage(&FragmentBRDF);
	VoxelBRDFShader.Compile();
	
	ShaderProgram BRDFShader = ShaderProgram(L"BRDF");
	BRDFShader.AppendStage(&VertexBase);
	BRDFShader.AppendStage(&FragmentBRDF);
	BRDFShader.Compile();

	ShaderProgram UnlitShader = ShaderProgram(L"UnLit");
	UnlitShader.AppendStage(&VertexBase);
	UnlitShader.AppendStage(&FragmentUnlit);
	UnlitShader.Compile();

	ShaderProgram RenderTextureShader = ShaderProgram(L"RenderTexture");
	RenderTextureShader.AppendStage(&PassthroughVertex);
	RenderTextureShader.AppendStage(&FragRenderTexture);
	RenderTextureShader.Compile();

	Material BaseMaterial = Material();
	BaseMaterial.SetShaderProgram(&BRDFShader);

	Material VoxelizeMaterial = Material();
	VoxelizeMaterial.SetShaderProgram(&VoxelBRDFShader);

	Material UnlitMaterial = Material();
	UnlitMaterial.SetShaderProgram(&UnlitShader);

	Material RenderTexMaterial = Material();
	RenderTexMaterial.DepthFunction = Graphics::DepthFunction::Always;
	RenderTexMaterial.CullMode = Graphics::CullMode::None;
	RenderTexMaterial.SetShaderProgram(&RenderTextureShader);

	Texture3D VoxelizedSceneTexture = Texture3D(1 << 7, Graphics::RGBA, Graphics::MinMagNearest, Graphics::Clamp);
	Texture2D RenderedTexture = Texture2D({ MainWindow->GetWidth(), MainWindow->GetHeight() }, Graphics::RGB, Graphics::MinLinearMagNearest, Graphics::Repeat);
	RenderTarget Framebuffer = RenderTarget({ MainWindow->GetWidth(), MainWindow->GetHeight() }, &RenderedTexture, true);

	float MaterialMetalness = 0.F;
	float MaterialRoughness = 0.54F;
	float LightIntencity = 100.F;

	TArray<Matrix4x4> Matrices;
	Matrices.push_back(Matrix4x4());

	///////// Create Matrices Buffer //////////////
	GLuint ModelMatrixBuffer;
	glGenBuffers(1, &ModelMatrixBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);

	srand((unsigned int)glfwGetTime());

	///////// Get Locations from Shaders /////////////
	// --- Get the ID of the uniforms
	GLuint       ModelMatrixLocation = BRDFShader.GetAttribLocation("_iModelMatrix");

	GLuint         Texture3DLocation = VoxelBRDFShader.GetUniformLocation("_Texture3D");

	///////////////////////////////////////////////////

	std::vector<MeshFaces> Faces; std::vector<MeshVertices> Vertices;
	MeshLoader::FromOBJ(FileManager::Open(L"Data\\Models\\Sponza.obj"), &Faces, &Vertices, false);
	std::vector<Mesh> OBJModels;
	float MeshSelector = 0;
	for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
		OBJModels.push_back(Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount]));
	}

	MeshLoader::FromOBJ(FileManager::Open(L"Data\\Models\\Quad.obj"), &Faces, &Vertices, false);
	std::vector<Mesh> QuadModels;;
	for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
		QuadModels.push_back(Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount]));
	}

	MeshLoader::FromOBJ(FileManager::Open(L"Data\\Models\\Sphere.obj"), &Faces, &Vertices, true);
	std::vector<Mesh> LightModels;
	for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
		LightModels.push_back(Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount]));
	}

	MeshVertex* Positions = (MeshVertex*)malloc(OBJModels[0].Vertices.size() * sizeof(MeshVertex));
	unsigned long InputTimeSum = 0;

	do {
		Time::Tick();

		ProjectionMatrix = Matrix4x4::Perspective(
			45.0F * MathConstants::DegreeToRad,	// Aperute angle
			MainWindow->AspectRatio(),	        // Aspect ratio
			0.1F,						        // Near plane
			200.0F						        // Far plane
		);

		// --- Camera rotation, position Matrix
		Vector2 CursorPosition = MainWindow->GetMousePosition();
		
		ViewOrientation = { CursorPosition.y * 0.01F, CursorPosition.x * 0.01F, 0.F };
		Quaternion FrameRotation = Quaternion(ViewOrientation.x, { 1, 0, 0 });
		           FrameRotation *= Quaternion(ViewOrientation.y, { 0, 1, 0 });
		
		if (MainWindow->GetKeyDown(GLFW_KEY_W)) {
			Vector3 Forward = FrameRotation.ToMatrix4x4() * Vector3(0, 0, ViewSpeed);
			EyePosition += Forward * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_A)) {
			Vector3 Right = FrameRotation.ToMatrix4x4() * Vector3(ViewSpeed, 0, 0);
			EyePosition += Right * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_S)) {
			Vector3 Back = FrameRotation.ToMatrix4x4() * Vector3(0, 0, -ViewSpeed);
			EyePosition += Back * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_D)) {
			Vector3 Left = FrameRotation.ToMatrix4x4() * Vector3(-ViewSpeed, 0, 0);
			EyePosition += Left * Time::GetDeltaTime();
		}
		ViewMatrix =
			Matrix4x4::Translate(EyePosition) * FrameRotation.ToMatrix4x4();

		if (MainWindow->GetKeyDown(GLFW_KEY_N)) {
			MaterialMetalness -= 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_M)) {
			MaterialMetalness += 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_E)) {
			MaterialRoughness -= 0.1F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_R)) {
			MaterialRoughness += 0.1F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_L)) {
			LightIntencity += 10 * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_K)) {
			LightIntencity -= 10 * Time::GetDeltaTime();
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_LEFT_SHIFT)) {
			if (MainWindow->GetKeyDown(GLFW_KEY_W)) {
				if (BaseMaterial.RenderMode == Graphics::RenderMode::Fill) {
					BaseMaterial.RenderMode = Graphics::RenderMode::Wire;
				} else {
					BaseMaterial.RenderMode = Graphics::RenderMode::Fill;
				}
			}
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_UP)) {
			MeshSelector += Time::GetDeltaTime() * 10;
			MeshSelector = MeshSelector > OBJModels.size() - 1 ? OBJModels.size() - 1 : MeshSelector;
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_DOWN)) {
			MeshSelector -= Time::GetDeltaTime() * 10;
			MeshSelector = MeshSelector < 0 ? 0 : MeshSelector;
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_U)) {
			InputTimeSum += Time::GetDeltaTimeMilis();
			if (InputTimeSum > (300)) {
				for (size_t i = 0; i < 1; i++) {
					Matrices.push_back(
						Matrix4x4::Translate({ ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128 })
					);
				}
			}
			if (MainWindow->GetKeyDown(GLFW_KEY_Q)) {
				for (size_t i = 0; i < 100; i++) {
					Matrices.push_back(
						Matrix4x4::Translate({ ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128 })
					);
				}
			}
		}

		// --- Draw the meshs(es) !
		RenderTimeSum += Time::GetDeltaTimeMilis();
		if (RenderTimeSum > (1000 / 60)) {
			RenderTimeSum = 0;
			size_t TriangleCount = 0;
			size_t VerticesCount = 0;

			glViewport(0, 0, MainWindow->GetWidth(), MainWindow->GetHeight());
			MainWindow->ClearWindow();
			Framebuffer.Clear();
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			Debug::Timer Timer;
			Timer.Start();

			// Debug::Log(Debug::LogDebug, L"Test Host Vector[%d] %s", 0, OBJModels[0].Vertices[0].Position.ToString().c_str());

			// std::memcpy(Positions, &OBJModels[0].Vertices[0], OBJModels[0].Vertices.size() * sizeof(MeshVertex));

			// --- Run the device part of the program
			bool bTestResult = false;
			bTestResult = RayTracingTexture2D(&RenderedTexture);
			// bTestResult = FindBoundingBox((int)OBJModels[0].Vertices.size(), &OBJModels[0].Vertices[0]);
			// bTestResult = VoxelizeToTexture3D(VoxelizedSceneTexture, (int)OBJModels[0].Vertices.size(), &OBJModels[0].Vertices[0]);

			// Debug::Log(Debug::LogDebug, L"Test Device Vector[%d] %s", 0, Positions[0].Position.ToString().c_str());
	
			Timer.Stop();
			// Debug::Log(
			// 	Debug::LogWarning, L"CUDA Test with %s elements durantion: %dms",
			// 	Text::FormatUnit(OBJModels[0].Vertices.size(), 2).c_str(),
			// 	Timer.GetEnlapsed()
			// );
			// Debug::Log(Debug::NoLog, L"\n");

			// Framebuffer.Use();
			BaseMaterial.Use();
			
			BaseMaterial.SetFloat3Array( "_ViewPosition",            EyePosition.PointerToValue() );
			BaseMaterial.SetFloat3Array( "_Lights[0].Position",    LightPosition.PointerToValue() );
			BaseMaterial.SetFloat3Array( "_Lights[0].Color",          Vector3(1).PointerToValue() );
			BaseMaterial.SetFloat1Array( "_Lights[0].Intencity",                  &LightIntencity );
			BaseMaterial.SetFloat3Array( "_Lights[1].Position", (-LightPosition).PointerToValue() );
			BaseMaterial.SetFloat3Array( "_Lights[1].Color",          Vector3(1).PointerToValue() );
			BaseMaterial.SetFloat1Array( "_Lights[1].Intencity",                  &LightIntencity );
			BaseMaterial.SetFloat1Array( "_Material.Metalness",                &MaterialMetalness );
			BaseMaterial.SetFloat1Array( "_Material.Roughness",                &MaterialRoughness );
			BaseMaterial.SetFloat3Array( "_Material.Color", Vector3(.6F, .2F, 0).PointerToValue() );
			
			BaseMaterial.SetMatrix4x4Array( "_ProjectionMatrix", ProjectionMatrix.PointerToValue() );
			BaseMaterial.SetMatrix4x4Array( "_ViewMatrix",             ViewMatrix.PointerToValue() );

			for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)OBJModels.size(); ++MeshCount) {
				OBJModels[MeshCount].BindVertexArray();

				BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", (int)Matrices.size(), &Matrices[0], ModelMatrixBuffer);

				OBJModels[MeshCount].DrawInstanciated((GLsizei)Matrices.size());
				TriangleCount += OBJModels[MeshCount].Faces.size() * Matrices.size();
				VerticesCount += OBJModels[MeshCount].Vertices.size() * Matrices.size();
			}

			UnlitMaterial.Use();

			UnlitMaterial.SetMatrix4x4Array( "_ProjectionMatrix", ProjectionMatrix.PointerToValue() );
			UnlitMaterial.SetMatrix4x4Array( "_ViewMatrix",             ViewMatrix.PointerToValue() );
			UnlitMaterial.SetFloat3Array( "_ViewPosition",             EyePosition.PointerToValue() );
			UnlitMaterial.SetFloat3Array( "_Material.Color",            Vector3(1).PointerToValue() );
			
			LightModels[0].BindVertexArray();

			vector<Matrix4x4> LightPositions;
			LightPositions.push_back(Matrix4x4::Scale(0.1F) * Matrix4x4::Translate(LightPosition));
			LightPositions.push_back(Matrix4x4::Scale(0.1F) * Matrix4x4::Translate(-LightPosition));

			UnlitMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 2, &LightPositions[0], ModelMatrixBuffer);

			LightModels[0].DrawInstanciated(2);

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glViewport(0, 0, MainWindow->GetWidth(), MainWindow->GetHeight());

			RenderTexMaterial.Use();

			float AppTime = (float)Time::GetApplicationTime() / 1000;
			RenderTexMaterial.SetFloat1Array("_Time", &AppTime);
			RenderTexMaterial.SetFloat2Array("_MainTextureSize", RenderedTexture.GetDimension().FloatVector2().PointerToValue());
			RenderTexMaterial.SetTexture2D("_MainTexture", &RenderedTexture, 0);

			QuadModels[0].BindVertexArray();

			Matrix4x4 QuadPosition = Matrix4x4::Translate({ 0, 0, 0 });
			RenderTexMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer);

			QuadModels[0].DrawInstanciated(1);

			MainWindow->SetWindowName(
				Text::Formatted(L"%s - Temp(%d°), FPS(%.0f)(%.2f ms), Instances(%s), Vertices(%s), Triangles(%s), Camera(P%s, R%s)",
					L"EmptySource",
					Debug::GetDeviceTemperature(0),
					Time::GetFrameRate(),
					(1 / Time::GetFrameRate()) * 1000,
					Text::FormatUnit(Matrices.size(), 2).c_str(),
					Text::FormatUnit(VerticesCount, 2).c_str(),
					Text::FormatUnit(TriangleCount, 2).c_str(),
					Text::FormatMath(EyePosition).c_str(),
					Text::FormatMath(Math::ClampAngleComponents(FrameRotation.ToEulerAngles() * MathConstants::RadToDegree)).c_str()
				)
			);

			MainWindow->EndOfFrame();
			glBindVertexArray(0);
		}

		MainWindow->PollEvents();


	} while (
		MainWindow->ShouldClose() == false && !MainWindow->GetKeyDown(GLFW_KEY_ESCAPE)
	);

	delete[] Positions;
}