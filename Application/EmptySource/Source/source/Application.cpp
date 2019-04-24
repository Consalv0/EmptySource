﻿#include "../include/Core.h"
#include "../include/Math/CoreMath.h"
#include "../include/Application.h"
#ifndef __APPLE__
#include "../include/CoreCUDA.h"
#endif

#include "../include/Utility/LogGraphics.h"
#include "../include/Utility/DeviceFunctions.h"
#include "../include/Utility/TextFormattingMath.h"
#include "../include/Utility/Timer.h"

#include "../include/CoreTime.h"
#include "../include/Window.h"
#include "../include/FileManager.h"
#include "../include/Mesh.h"
#include "../include/MeshLoader.h"
#include "../include/Material.h"
#include "../include/ShaderProgram.h"
#include "../include/Space.h"
#include "../include/Object.h"
#include "../include/Transform.h"

#include "../include/Font.h"
#include "../include/Text2DGenerator.h"
#include "../include/Graphics.h"
#include "../include/RenderTarget.h"
#include "../include/Texture2D.h"
#include "../include/Texture3D.h"
#include "../include/Cubemap.h"
#include "../include/ImageLoader.h"

#include <thread>

// int FindBoundingBox(int N, MeshVertex * Vertices);
// int VoxelizeToTexture3D(Texture3D * Texture, int N, MeshVertex * Vertices);
int RTRenderToTexture2D(Texture2D * Texture, std::vector<Vector4> * Spheres, const void * dRandState);
const void * GetRandomArray(IntVector2 Dimension);

ApplicationWindow* CoreApplication::MainWindow = NULL;
bool CoreApplication::bInitialized = false;
double CoreApplication::RenderTimeSum = 0;

bool CoreApplication::InitalizeGLAD() {
	if (!gladLoadGL()) {
		Debug::Log(Debug::LogCritical, L"Unable to load OpenGL functions!");
		return false;
	}

#ifndef __APPLE__
	glEnable(GL_DEBUG_OUTPUT);
	// glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(Debug::OGLError, nullptr);
	// --- Enable all messages, all sources, all levels, and all IDs:
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
#endif

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
#ifdef __APPLE__
    if (InitializeGLFW(4, 1) == false) return;
#else
    if (InitializeGLFW(4, 6) == false) return;
#endif
	if (InitializeWindow() == false) return;
	if (InitalizeGLAD() == false) return;
    if (Debug::InitializeDeviceFunctions() == false) {
        Debug::Log(Debug::LogWarning, L"Couldn't initialize device functions");
    };

#ifdef WIN32
	CUDA::FindCudaDevice();
#endif

	bInitialized = true;
}

void CoreApplication::Close() {
	if (MainWindow) {
		MainWindow->Terminate();
		delete MainWindow;
	}

	glfwTerminate();
    Debug::CloseDeviceFunctions();
};

void CoreApplication::MainLoop() {
	if (!bInitialized) return;

	Space NewSpace; Space OtherNewSpace(NewSpace);
	Object* GObject = Space::GetFirstSpace()->MakeObject();
	GObject->Delete();

	glfwSwapInterval(0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	Font FontFace;
	Font::InitializeFreeType();
	FontFace.Initialize(FileManager::Open(L"Resources/Fonts/SourceSansPro.ttf"));
 
	Text2DGenerator TextGenerator;
	TextGenerator.TextFont = &FontFace;
	TextGenerator.GlyphHeight = 32;
	TextGenerator.AtlasSize = 512;
	TextGenerator.PixelRange = 1.5F;
	TextGenerator.Pivot = 0;

	Bitmap<UCharRGBA> BaseAlbedo;
	Bitmap<UCharRed> BaseMetallic, BaseRoughness;
	ImageLoader::Load(BaseAlbedo, FileManager::Open(L"Resources/Textures/EscafandraMV1971_BaseColor.png"));
	BaseAlbedo.FlipVertically();
	ImageLoader::Load(BaseMetallic, FileManager::Open(L"Resources/Textures/EscafandraMV1971_Metallic.png"));
	BaseMetallic.FlipVertically();
	ImageLoader::Load(BaseRoughness, FileManager::Open(L"Resources/Textures/EscafandraMV1971_Roughness.png"));
	BaseRoughness.FlipVertically();
	Bitmap<UCharRGB> White, Black;
	ImageLoader::Load(White, FileManager::Open(L"Resources/Textures/White.jpg"));
	White.FlipVertically();
	ImageLoader::Load(Black, FileManager::Open(L"Resources/Textures/Black.jpg"));
	Black.FlipVertically();

	Bitmap<FloatRGB> Equirectangular;
	ImageLoader::Load(Equirectangular, FileManager::Open(L"Resources/Textures/Factory_Catwalk_2k.hdr"));
	Equirectangular.FlipVertically();

	Texture2D EquirectangularTexture = Texture2D(
		IntVector2(Equirectangular.GetWidth(), Equirectangular.GetHeight()),
		Graphics::CF_RGB16F,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_RGB16F,
		Equirectangular.PointerToValue()
	);
	Texture2D BaseAlbedoTexture = Texture2D(
		IntVector2(BaseAlbedo.GetWidth(), BaseAlbedo.GetHeight()),
		Graphics::CF_RGBA,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Border,
		Graphics::CF_RGBA,
		BaseAlbedo.PointerToValue()
	);
	BaseAlbedoTexture.GenerateMipMaps();
	Texture2D BaseMetallicTexture = Texture2D(
		IntVector2(BaseMetallic.GetWidth(), BaseMetallic.GetHeight()),
		Graphics::CF_Red,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_Red,
		BaseMetallic.PointerToValue()
	);
	BaseMetallicTexture.GenerateMipMaps();
	Texture2D BaseRoughnessTexture = Texture2D(
		IntVector2(BaseRoughness.GetWidth(), BaseRoughness.GetHeight()),
		Graphics::CF_Red,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_Red,
		BaseRoughness.PointerToValue()
	);
	BaseRoughnessTexture.GenerateMipMaps();
	Texture2D WhiteTexture = Texture2D(
		IntVector2(White.GetWidth(), White.GetHeight()),
		Graphics::CF_RGB,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_RGB,
		White.PointerToValue()
	);
	WhiteTexture.GenerateMipMaps();
	Texture2D BlackTexture = Texture2D(
		IntVector2(Black.GetWidth(), Black.GetHeight()),
		Graphics::CF_RGB,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_RGB,
		Black.PointerToValue()
	);
	BlackTexture.GenerateMipMaps();

	Bitmap<UCharRed> FontAtlas;
	TextGenerator.GenerateGlyphAtlas(FontAtlas);
    
	Texture2D FontMap = Texture2D(
		IntVector2(TextGenerator.AtlasSize),
		Graphics::CF_Red,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Border,
		Graphics::CF_Red,
		FontAtlas.PointerToValue()
	);
	FontMap.GenerateMipMaps();

	/////////// Creating MVP (ModelMatrix, ViewMatrix, Poryection) Matrix //////////////
	// --- Perpective matrix (ProjectionMatrix)
	Matrix4x4 ProjectionMatrix;

	Vector3 EyePosition = { -1.132F , 2.692F , -4.048F };
	Vector3 LightPosition0 = Vector3(2, 1);
	Vector3 LightPosition1 = Vector3(2, 1);
	// --- Camera rotation, position Matrix
	float ViewSpeed = 3;
	Vector3 ViewOrientation;
	Matrix4x4 ViewMatrix;

	// --- Create and compile our GLSL shader programs from text files
	ShaderStage EquirectangularToCubemapVert =
		ShaderStage(ShaderType::Vertex, FileManager::Open(L"Resources/Shaders/EquirectangularToCubemap.vertex.glsl"));
	ShaderStage EquirectangularToCubemapFrag =
		ShaderStage(ShaderType::Fragment, FileManager::Open(L"Resources/Shaders/EquirectangularToCubemap.fragment.glsl"));
	ShaderStage HDRClamping       = ShaderStage(ShaderType::Fragment, FileManager::Open(L"Resources/Shaders/HDRClamping.fragment.glsl"));
	ShaderStage VertexBase        = ShaderStage(ShaderType::Vertex,   FileManager::Open(L"Resources/Shaders/Base.vertex.glsl"));
	ShaderStage VoxelizerVertex   = ShaderStage(ShaderType::Vertex,   FileManager::Open(L"Resources/Shaders/Voxelizer.vertex.glsl"));
	ShaderStage PassthroughVertex = ShaderStage(ShaderType::Vertex,   FileManager::Open(L"Resources/Shaders/Passthrough.vertex.glsl"));
	ShaderStage FragmentBRDF      = ShaderStage(ShaderType::Fragment, FileManager::Open(L"Resources/Shaders/BRDF.fragment.glsl"));
	ShaderStage IntegrateBRDF     = ShaderStage(ShaderType::Fragment, FileManager::Open(L"Resources/Shaders/IntegrateBRDF.fragment.glsl"));
	ShaderStage FragRenderTexture = ShaderStage(ShaderType::Fragment, FileManager::Open(L"Resources/Shaders/RenderTexture.fragment.glsl"));
	ShaderStage FragRenderText    = ShaderStage(ShaderType::Fragment, FileManager::Open(L"Resources/Shaders/RenderText.fragment.glsl"));
	ShaderStage FragRenderCubemap = ShaderStage(ShaderType::Fragment, FileManager::Open(L"Resources/Shaders/RenderCubemap.fragment.glsl"));
	ShaderStage FragmentUnlit     = ShaderStage(ShaderType::Fragment, FileManager::Open(L"Resources/Shaders/Unlit.fragment.glsl"));
	ShaderStage Voxelizer         = ShaderStage(ShaderType::Geometry, FileManager::Open(L"Resources/Shaders/Voxelizer.geometry.glsl"));

	ShaderProgram EquirectangularToCubemapShader = ShaderProgram(L"EquirectangularToCubemap");
	EquirectangularToCubemapShader.AppendStage(&EquirectangularToCubemapVert);
	EquirectangularToCubemapShader.AppendStage(&EquirectangularToCubemapFrag);
	EquirectangularToCubemapShader.Compile();

	ShaderProgram HDRClampingShader = ShaderProgram(L"HDRClampingShader");
	HDRClampingShader.AppendStage(&PassthroughVertex);
	HDRClampingShader.AppendStage(&HDRClamping);
	HDRClampingShader.Compile();

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

	ShaderProgram IntegrateBRDFShader = ShaderProgram(L"IntegrateBRDF");
	IntegrateBRDFShader.AppendStage(&PassthroughVertex);
	IntegrateBRDFShader.AppendStage(&IntegrateBRDF);
	IntegrateBRDFShader.Compile();
    
	ShaderProgram RenderTextShader = ShaderProgram(L"RenderText");
	RenderTextShader.AppendStage(&PassthroughVertex);
	RenderTextShader.AppendStage(&FragRenderText);
	RenderTextShader.Compile();

	ShaderProgram RenderCubemapShader = ShaderProgram(L"RenderCubemap");
	RenderCubemapShader.AppendStage(&VertexBase);
	RenderCubemapShader.AppendStage(&FragRenderCubemap);
	RenderCubemapShader.Compile();

	Material BaseMaterial = Material();
	BaseMaterial.SetShaderProgram(&BRDFShader);
    
    Material VoxelizeMaterial = Material();
	VoxelizeMaterial.SetShaderProgram(&VoxelBRDFShader);
    
	Material UnlitMaterial = Material();
	UnlitMaterial.SetShaderProgram(&UnlitShader);
    
	Material RenderTextureMaterial = Material();
	RenderTextureMaterial.DepthFunction = Graphics::DF_Always;
	RenderTextureMaterial.CullMode = Graphics::CM_None;
	RenderTextureMaterial.SetShaderProgram(&RenderTextureShader);
    
	Material RenderTextMaterial = Material();
	RenderTextMaterial.DepthFunction = Graphics::DF_Always;
	RenderTextMaterial.CullMode = Graphics::CM_None;
	RenderTextMaterial.SetShaderProgram(&RenderTextShader);

	Material RenderCubemapMaterial = Material();
	RenderCubemapMaterial.CullMode = Graphics::CM_None;
	RenderCubemapMaterial.SetShaderProgram(&RenderCubemapShader);

	Material IntegrateBRDFMaterial = Material();
	IntegrateBRDFMaterial.DepthFunction = Graphics::DF_Always;
	IntegrateBRDFMaterial.CullMode = Graphics::CM_None;
	IntegrateBRDFMaterial.SetShaderProgram(&IntegrateBRDFShader);

	Material HDRClampingMaterial = Material();
	HDRClampingMaterial.DepthFunction = Graphics::DF_Always;
	HDRClampingMaterial.CullMode = Graphics::CM_None;
	HDRClampingMaterial.SetShaderProgram(&HDRClampingShader);

	srand((unsigned int)glfwGetTime());
	TArray<Mesh> OBJModels;
	TArray<Mesh> LightModels;
	Mesh CubeModel;
	Mesh QuadModel;
	float MeshSelector = 0;

	TArray<std::thread> Threads;
	Threads.push_back(std::thread([&OBJModels]() {
		TArray<MeshFaces> Faces; TArray<MeshVertices> Vertices;
		OBJLoader::Load(FileManager::Open(L"Resources/Models/Escafandra.obj"), &Faces, &Vertices, false);
		for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
			OBJModels.push_back(Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount]));
		}
	}));
	// Threads.push_back(std::thread([&OBJModels]() {
	// 	TArray<MeshFaces> Faces; TArray<MeshVertices> Vertices;
	// 	OBJLoader::Load(FileManager::Open(L"Resources/Models/Escafandra.obj"), &Faces, &Vertices, false);
	// 	for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
	// 		OBJModels.push_back(Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount]));
	// 	}
	// }));

	Threads.push_back(std::thread([&LightModels]() {
		TArray<MeshFaces> Faces; TArray<MeshVertices> Vertices;
		OBJLoader::Load(FileManager::Open(L"Resources/Models/Monkey.obj"), &Faces, &Vertices, true);
		for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
			LightModels.push_back(Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount]));
		}
	}));

	Threads.push_back(std::thread([&CubeModel]() {
		TArray<MeshFaces> Faces; TArray<MeshVertices> Vertices;
		OBJLoader::Load(FileManager::Open(L"Resources/Models/Sphere.obj"), &Faces, &Vertices, true);
		CubeModel = Mesh(&Faces[0], &Vertices[0]);
	}));
    
	Texture2D RenderedTexture = Texture2D(
	 	IntVector2(MainWindow->GetWidth(), MainWindow->GetHeight()) / 2, Graphics::CF_RGBA32F, Graphics::FM_MinLinearMagNearest, Graphics::AM_Repeat
	);

	///////// Create Matrices Buffer //////////////
	GLuint ModelMatrixBuffer;
	glGenBuffers(1, &ModelMatrixBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);

	TArray<MeshFaces> Faces; TArray<MeshVertices> Vertices;
	OBJLoader::Load(FileManager::Open(L"Resources/Models/Quad.obj"), &Faces, &Vertices, false);
	QuadModel = (Mesh(&Faces[0], &Vertices[0]));

	Texture2D EquirectangularTextureHDR = Texture2D(
		IntVector2(Equirectangular.GetWidth(), Equirectangular.GetHeight()), Graphics::CF_RGB16F, Graphics::FM_MinMagLinear, Graphics::AM_Repeat
	);
	{
		RenderTarget Renderer = RenderTarget();
		Renderer.SetUpBuffers();
		EquirectangularTextureHDR.Use();
		HDRClampingMaterial.Use();
		HDRClampingMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
		HDRClampingMaterial.SetTexture2D("_EquirectangularMap", &EquirectangularTexture, 0);
		Renderer.Resize(EquirectangularTextureHDR.GetWidth(), EquirectangularTextureHDR.GetHeight());
		QuadModel.SetUpBuffers();
		QuadModel.BindVertexArray();
		Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
		HDRClampingMaterial.SetAttribMatrix4x4Array(
			"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
		);

		Renderer.PrepareTexture(&EquirectangularTextureHDR);
		Renderer.Clear();
		QuadModel.DrawInstanciated(1);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		Renderer.Delete();
		EquirectangularTextureHDR.GenerateMipMaps();
	}

	Texture2D BRDFLut(IntVector2(512), Graphics::CF_RG16F, Graphics::FM_MinMagLinear, Graphics::AM_Clamp);
	{
		RenderTarget Renderer = RenderTarget();
		Renderer.SetUpBuffers();
		Renderer.Use();
		IntegrateBRDFMaterial.Use();
		IntegrateBRDFMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
		Renderer.Resize(BRDFLut.GetWidth(), BRDFLut.GetHeight());
		QuadModel.SetUpBuffers();
		QuadModel.BindVertexArray();
		Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
		IntegrateBRDFMaterial.SetAttribMatrix4x4Array(
			"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
		);

		Renderer.PrepareTexture(&BRDFLut);
		Renderer.Clear();
		QuadModel.DrawInstanciated(1);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		Renderer.Delete();
		// BRDFLut.GenerateMipMaps();
	}

	Cubemap CubemapTexture(Equirectangular.GetHeight() / 2, Graphics::CF_RGB16F, Graphics::FM_MinMagLinear, Graphics::AM_Clamp);
	{
		Mesh CubeModel;
		TArray<MeshFaces> Faces; TArray<MeshVertices> Vertices;
		OBJLoader::Load(FileManager::Open(L"Resources/Models/Cube.obj"), &Faces, &Vertices, false);
		for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
			CubeModel = Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount]);
		}
		Cubemap::FromHDREquirectangular(CubemapTexture, &EquirectangularTextureHDR, &CubeModel, &EquirectangularToCubemapShader);
	}

	float MaterialMetalness = 1.F;
	float MaterialRoughness = 1.F;
	float LightIntencity = 20.F;

	TArray<Transform> Transforms;
	Transforms.push_back(Transform());

	float MultiuseValue = 1;
	const int TextCount = 3;
	float FontSize = 14;
	float FontBoldness = 0.55F;
	WString RenderingText[TextCount];
	Mesh DynamicMesh;
	Point2 TextPivot;

	bool bRandomArray = false;
	const void* curandomStateArray = 0;

	glEnable(GL_BLEND);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	do {
		Time::Tick();

		ProjectionMatrix = Matrix4x4::Perspective(
			60.0F * MathConstants::DegreeToRad,	// Aperute angle
			MainWindow->AspectRatio(),	        // Aspect ratio
			0.03F,						        // Near plane
			1000.0F						        // Far plane
		);

		// --- Camera rotation, position Matrix
		Vector2 CursorPosition = MainWindow->GetMousePosition();
		
		Quaternion FrameRotation  = Quaternion::EulerAngles(Vector3(CursorPosition.y, -CursorPosition.x));

		if (MainWindow->GetKeyDown(GLFW_KEY_W)) {
			Vector3 Forward = FrameRotation * Vector3(0, 0, ViewSpeed);
			EyePosition += Forward * Time::GetDeltaTime() * (!MainWindow->GetKeyDown(GLFW_KEY_LEFT_SHIFT) ? 1.F : 4.F);
			// TextPivot.y += (FontSize + 100) * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_A)) {
			Vector3 Right = FrameRotation * Vector3(ViewSpeed, 0, 0);
			EyePosition += Right * Time::GetDeltaTime() * (!MainWindow->GetKeyDown(GLFW_KEY_LEFT_SHIFT) ? 1.F : 4.F);
			// TextPivot.x -= (FontSize + 100) * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_S)) {
			Vector3 Back = FrameRotation * Vector3(0, 0, -ViewSpeed);
			EyePosition += Back * Time::GetDeltaTime() * (!MainWindow->GetKeyDown(GLFW_KEY_LEFT_SHIFT) ? 1.F : 4.F);
			// TextPivot.y -= (FontSize + 100) * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_D)) {
			Vector3 Left = FrameRotation * Vector3(-ViewSpeed, 0, 0);
			EyePosition += Left * Time::GetDeltaTime() * (!MainWindow->GetKeyDown(GLFW_KEY_LEFT_SHIFT) ? 1.F : 4.F);
			// TextPivot.x += (FontSize + 100) * Time::GetDeltaTime();
		}
		
		ViewMatrix = Matrix4x4::LookAt(EyePosition, EyePosition + FrameRotation * Vector3(0, 0, 1), FrameRotation * Vector3(0, 1));

		if (MainWindow->GetKeyDown(GLFW_KEY_N)) {
			MaterialMetalness -= 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_M)) {
			MaterialMetalness += 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_E)) {
			MaterialRoughness -= 0.5F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_R)) {
			MaterialRoughness += 0.5F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_L)) {
			LightIntencity += LightIntencity * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_K)) {
			LightIntencity -= LightIntencity * Time::GetDeltaTime();
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_LEFT_SHIFT)) {
		if (MainWindow->GetKeyDown(GLFW_KEY_W)) {
				if (BaseMaterial.RenderMode == Graphics::RM_Fill) {
					BaseMaterial.RenderMode = Graphics::RM_Wire;
				} else {
					BaseMaterial.RenderMode = Graphics::RM_Fill;
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
		if(MainWindow->GetKeyDown(GLFW_KEY_RIGHT)) {
			MultiuseValue += Time::GetDeltaTime() * MultiuseValue;
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_LEFT)) {
			MultiuseValue -= Time::GetDeltaTime() * MultiuseValue;
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_LEFT_SHIFT)) {
			if (MainWindow->GetKeyDown(GLFW_KEY_I)) {
				FontSize += Time::GetDeltaTime() * FontSize;
			}

			if (MainWindow->GetKeyDown(GLFW_KEY_K)) {
				FontSize -= Time::GetDeltaTime() * FontSize;
			}
		} else {
			if (MainWindow->GetKeyDown(GLFW_KEY_I)) {
				FontBoldness += Time::GetDeltaTime() / 10;
			}

			if (MainWindow->GetKeyDown(GLFW_KEY_K)) {
				FontBoldness -= Time::GetDeltaTime() / 10;
			}
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_V)) {
			for (int i = 0; i < 10; i++) {
				RenderingText[1] += (unsigned long)(rand() % 0x3ff);
			}
		}

		for (int i = 0; i < TextCount; i++) {
			if (TextGenerator.FindCharacters(RenderingText[i]) > 0) {
				TextGenerator.GenerateGlyphAtlas(FontAtlas);
				FontMap.Delete();
				FontMap = Texture2D(
					IntVector2(TextGenerator.AtlasSize),
					Graphics::CF_Red,
					Graphics::FM_MinMagLinear,
					Graphics::AM_Border,
					Graphics::CF_Red,
					FontAtlas.PointerToValue()
				);
				FontMap.GenerateMipMaps();
			}
		}

		Transforms[0].Rotation = Quaternion::AxisAngle(Vector3(0, 1, 0).Normalized(), Time::GetDeltaTime() * 0.4F) * Transforms[0].Rotation;

		RenderTimeSum += Time::GetDeltaTime();
		const float MaxFramerate = (1 / 65.F);
		if (RenderTimeSum > MaxFramerate) {
			RenderTimeSum = 0;
			size_t TriangleCount = 0;
			size_t VerticesCount = 0;

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glViewport(0, 0, MainWindow->GetWidth(), MainWindow->GetHeight());
			MainWindow->ClearWindow();
		
			Debug::Timer Timer;

			// --- Run the device part of the program
			if (!bRandomArray) {
				// curandomStateArray = GetRandomArray(RenderedTexture.GetDimension());
				bRandomArray = true;
			}
			bool bTestResult = false;

			LightPosition0 = Transforms[0].Position + (Transforms[0].Rotation * Vector3(0, 0, 4));
			LightPosition1 = Vector3();

			TArray<Vector4> Spheres = TArray<Vector4>();
			Spheres.push_back(Vector4(Matrix4x4::Translation(Vector3(0, 0, -1.F)) * Vector4(0.F, 0.F, 0.F, 1.F), 0.25F));
			Spheres.push_back(Vector4((Matrix4x4::Translation(Vector3(0, 0, -1.F)) * Quaternion::EulerAngles(
				Vector3(MainWindow->GetMousePosition().x, MainWindow->GetMousePosition().y, 0)
			).ToMatrix4x4() * Matrix4x4::Translation(Vector3(0.5F))) * Vector4(0.F, 0.F, 0.F, 1.F), 0.5F));
			// bTestResult = RTRenderToTexture2D(&RenderedTexture, &Spheres, curandomStateArray);

			// Framebuffer.Use();

			RenderCubemapMaterial.Use();

			float MaterialRoughnessTemp = MaterialRoughness * CubemapTexture.GetMipmapCount();
			RenderCubemapMaterial.SetMatrix4x4Array("_ProjectionMatrix", ProjectionMatrix.PointerToValue());
			RenderCubemapMaterial.SetMatrix4x4Array("_ViewMatrix", ViewMatrix.PointerToValue());
			RenderCubemapMaterial.SetTextureCubemap("_Skybox", &CubemapTexture, 0);
			RenderCubemapMaterial.SetFloat1Array("_Roughness", &MaterialRoughnessTemp);

			if (CubeModel.Faces.size() >= 1) {
				CubeModel.SetUpBuffers();
				CubeModel.BindVertexArray();
                
                Matrix4x4 MatrixScale = Matrix4x4::Scaling({ 500, 500, 500 });
				RenderCubemapMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 2, MatrixScale.PointerToValue(), ModelMatrixBuffer);
				CubeModel.DrawElement();
			}

			BaseMaterial.Use();
			
			BaseMaterial.SetFloat3Array( "_ViewPosition",            EyePosition.PointerToValue() );
			BaseMaterial.SetFloat3Array( "_Lights[0].Position",   LightPosition0.PointerToValue() );
			BaseMaterial.SetFloat3Array( "_Lights[0].Color",    Vector3(0, 0, 1).PointerToValue() );
			BaseMaterial.SetFloat1Array( "_Lights[0].Intencity",                  &LightIntencity );
			BaseMaterial.SetFloat3Array( "_Lights[1].Position",   LightPosition1.PointerToValue() );
			BaseMaterial.SetFloat3Array( "_Lights[1].Color",    Vector3(1, 0, 0).PointerToValue() );
			BaseMaterial.SetFloat1Array( "_Lights[1].Intencity",                  &LightIntencity );
			BaseMaterial.SetFloat1Array( "_Material.Metalness",                &MaterialMetalness );
			BaseMaterial.SetFloat1Array( "_Material.Roughness",                &MaterialRoughness );
			BaseMaterial.SetFloat3Array( "_Material.Color",         Vector3(1.F).PointerToValue() );

			BaseMaterial.SetMatrix4x4Array( "_ProjectionMatrix",       ProjectionMatrix.PointerToValue() );
			BaseMaterial.SetMatrix4x4Array( "_ViewMatrix",                   ViewMatrix.PointerToValue() );
			BaseMaterial.SetTexture2D("_MainTexture", &BaseAlbedoTexture, 0);
			BaseMaterial.SetTexture2D("_RoughnessTexture", &BaseRoughnessTexture, 1);
			BaseMaterial.SetTexture2D("_MetallicTexture", &BaseMetallicTexture, 2);
			BaseMaterial.SetTexture2D("_BRDFLUT", &BRDFLut, 3);
			BaseMaterial.SetTextureCubemap("_EnviromentMap", &CubemapTexture, 4);
			float CubemapTextureMipmaps = CubemapTexture.GetMipmapCount();
			BaseMaterial.SetFloat1Array("_EnviromentMapLods", &CubemapTextureMipmaps);

			// Transforms[0].Position += Transforms[0].Rotation * Vector3(0, 0, Time::GetDeltaTime() * 2);
			Matrix4x4 TransformMat = Transforms[0].GetWorldMatrix();
			
			for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)OBJModels.size(); ++MeshCount) {
				OBJModels[MeshCount].SetUpBuffers();
				OBJModels[MeshCount].BindVertexArray();

				BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, &(TransformMat), ModelMatrixBuffer);

				OBJModels[MeshCount].DrawInstanciated((GLsizei)1);
				TriangleCount += OBJModels[MeshCount].Faces.size() * 1;
				VerticesCount += OBJModels[MeshCount].Vertices.size() * 1;
			}
			CubeModel.SetUpBuffers();
			CubeModel.BindVertexArray();

			BaseMaterial.SetTexture2D("_MainTexture", &WhiteTexture, 0);
			BaseMaterial.SetTexture2D("_RoughnessTexture", &WhiteTexture, 1);
			BaseMaterial.SetTexture2D("_MetallicTexture", &WhiteTexture, 2);
            Matrix4x4 ModelMatrix = (Matrix4x4::Translation(Vector3(0, 2, 0)));
			BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, &ModelMatrix, ModelMatrixBuffer);

			CubeModel.DrawInstanciated((GLsizei)1);
			TriangleCount += CubeModel.Faces.size() * 1;
			VerticesCount += CubeModel.Vertices.size() * 1;

			UnlitMaterial.Use();
            
			UnlitMaterial.SetMatrix4x4Array( "_ProjectionMatrix", ProjectionMatrix.PointerToValue() );
			UnlitMaterial.SetMatrix4x4Array( "_ViewMatrix",             ViewMatrix.PointerToValue() );
			UnlitMaterial.SetFloat3Array( "_ViewPosition",             EyePosition.PointerToValue() );
			UnlitMaterial.SetFloat3Array( "_Material.Color",            Vector3(1).PointerToValue() );
			
			if (LightModels.size() >= 1) {
				LightModels[0].SetUpBuffers();
				LightModels[0].BindVertexArray();

				TArray<Matrix4x4> LightPositions;
				LightPositions.push_back(Matrix4x4::Translation(LightPosition0) * Matrix4x4::Scaling(0.1F));
				LightPositions.push_back(Matrix4x4::Translation(LightPosition1) * Matrix4x4::Scaling(0.1F));

				UnlitMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 2, &LightPositions[0], ModelMatrixBuffer);

				LightModels[0].DrawInstanciated(2);
			}

			float AppTime = (float)Time::GetEpochTimeMicro() / 1000.F;
			glViewport(0, 0, EquirectangularTextureHDR.GetWidth() / 4, EquirectangularTextureHDR.GetHeight() / 4);
			
			RenderTextureMaterial.Use();

			RenderTextureMaterial.SetFloat1Array("_Time", &AppTime);
			RenderTextureMaterial.SetFloat2Array("_MainTextureSize", EquirectangularTextureHDR.GetDimension().FloatVector2().PointerToValue());
			RenderTextureMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
			RenderTextureMaterial.SetTexture2D("_MainTexture", &EquirectangularTextureHDR, 0);
			float LodLevel = log2f((float)EquirectangularTextureHDR.GetWidth()) * MultiuseValue;
			RenderTextureMaterial.SetFloat1Array("_Lod", &LodLevel);
			
			if (QuadModel.Faces.size() >= 1) {
				QuadModel.SetUpBuffers();
				QuadModel.BindVertexArray();

				Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
				RenderTextureMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1,
					/*(Quaternion({ MathConstants::HalfPi, 0, 0}).ToMatrix4x4() * */QuadPosition.PointerToValue(),
					ModelMatrixBuffer
				);

				QuadModel.DrawInstanciated(1);
			}

			glViewport(0, 0, MainWindow->GetWidth(), MainWindow->GetHeight());
			// --- Activate corresponding render state
			RenderTextMaterial.Use();
			RenderTextMaterial.SetFloat1Array("_Time", &AppTime);
			RenderTextMaterial.SetFloat2Array("_MainTextureSize", FontMap.GetDimension().FloatVector2().PointerToValue());
			RenderTextMaterial.SetMatrix4x4Array("_ProjectionMatrix",
				Matrix4x4::Orthographic(0.F, (float)MainWindow->GetWidth(), 0.F, (float)MainWindow->GetHeight()).PointerToValue()
			);

			RenderTextMaterial.SetTexture2D("_MainTexture", &FontMap, 0);
			RenderTextMaterial.SetFloat1Array("_TextSize", &FontSize);
			RenderTextMaterial.SetFloat1Array("_TextBold", &FontBoldness);

			double TimeCount = 0;
			int TotalCharacterSize = 0;
			DynamicMesh.Clear();
			for (int i = 0; i < TextCount; i++) {
				Timer.Start();
				TextGenerator.GenerateMesh(
					TextPivot + Vector2(0.F, MainWindow->GetHeight() - (i + 1) * FontSize + FontSize / TextGenerator.GlyphHeight),
					FontSize, RenderingText[i], &DynamicMesh.Faces, &DynamicMesh.Vertices
				);
				Timer.Stop();
				TimeCount += Timer.GetEnlapsedMili();
				TotalCharacterSize += (int)RenderingText[i].size();
			}
			if (DynamicMesh.SetUpBuffers()) {
				DynamicMesh.BindVertexArray();
				RenderTextMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
				DynamicMesh.DrawElement();
			}

			RenderingText[2] = Text::Formatted(
				L"Sphere Position(%ls), Sphere2 Position(%ls)",
				Text::FormatMath(Spheres[0]).c_str(),
				Text::FormatMath(Spheres[1]).c_str()
			);

			RenderingText[0] = Text::Formatted(
				L"Character(%.2f μs, %d), Temp [%.1f°], %.1f FPS (%.2f ms), Roughness(%.3f), Vertices(%ls), Triangles(%ls), MUV(%ls) Camera(P%ls, R%ls)",
				TimeCount / double(TotalCharacterSize) * 1000.0,
				TotalCharacterSize,
				Debug::GetDeviceTemperature(0),
				Time::GetFrameRatePerSecond(),
				(1.F / Time::GetFrameRatePerSecond()) * 1000.F,
				MaterialRoughness,
				Text::FormatUnit(VerticesCount, 2).c_str(),
				Text::FormatUnit(TriangleCount, 2).c_str(),
				Text::FormatUnit(MultiuseValue, 2).c_str(),
				Text::FormatMath(EyePosition).c_str(),
				Text::FormatMath(Math::ClampAngleComponents(FrameRotation.ToEulerAngles())).c_str()
			);

			MainWindow->EndOfFrame();
			glBindVertexArray(0);
		}

		MainWindow->PollEvents();


	} while (
		MainWindow->ShouldClose() == false && !MainWindow->GetKeyDown(GLFW_KEY_ESCAPE)
	);

	for (int i = 0; i < Threads.size(); i++) {
		Threads[i].join();
	}
	// delete[] Positions;
}
