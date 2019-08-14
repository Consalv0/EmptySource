
#include "CoreMinimal.h"
#include <imgui.h>
#include <examples/imgui_impl_sdl.h>
#include <examples/imgui_impl_opengl3.h>

#include "ImGui/ImGuiLayer.h"

// TEMPORARY
#include <SDL.h>
#include <GLAD/glad.h>

namespace EmptySource {

	ImGuiLayer::ImGuiLayer()
		: Layer(L"ImGuiLayer", 1000) {
	}

	void ImGuiLayer::OnAwake() {
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& IO = ImGui::GetIO(); (void)IO;
		IO.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
		// IO.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;     // Enable Gamepad Controls
		IO.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
		IO.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
		// IO.ConfigFlags |= ImGuiConfigFlags_ViewportsNoTaskBarIcons;
		// IO.ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge;

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();
		ImVec4* Colors = ImGui::GetStyle().Colors;
		Colors[ImGuiCol_WindowBg]				= ImVec4(0.06f, 0.06f, 0.06f, 0.53f);
		Colors[ImGuiCol_Text]					= ImVec4(0.99f, 0.96f, 0.93f, 1.00f);
		Colors[ImGuiCol_FrameBg]				= ImVec4(0.22f, 0.22f, 0.22f, 0.54f);
		Colors[ImGuiCol_FrameBgHovered]			= ImVec4(0.56f, 0.56f, 0.56f, 0.40f);
		Colors[ImGuiCol_FrameBgActive]			= ImVec4(0.65f, 0.65f, 0.65f, 0.67f);
		Colors[ImGuiCol_TitleBgActive]			= ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
		Colors[ImGuiCol_CheckMark]				= ImVec4(0.96f, 0.91f, 0.88f, 1.00f);
		Colors[ImGuiCol_SliderGrab]				= ImVec4(0.96f, 0.91f, 0.88f, 1.00f);
		Colors[ImGuiCol_SliderGrabActive]		= ImVec4(0.97f, 0.95f, 0.93f, 1.00f);
		Colors[ImGuiCol_Button]					= ImVec4(0.96f, 0.91f, 0.88f, 0.13f);
		Colors[ImGuiCol_ButtonHovered]			= ImVec4(0.96f, 0.91f, 0.88f, 0.32f);
		Colors[ImGuiCol_ButtonActive]			= ImVec4(0.96f, 0.91f, 0.98f, 0.40f);
		Colors[ImGuiCol_Header]					= ImVec4(0.39f, 0.39f, 0.39f, 0.31f);
		Colors[ImGuiCol_HeaderHovered]			= ImVec4(0.47f, 0.47f, 0.47f, 0.80f);
		Colors[ImGuiCol_HeaderActive]			= ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
		Colors[ImGuiCol_SeparatorHovered]		= ImVec4(0.82f, 0.82f, 0.82f, 0.78f);
		Colors[ImGuiCol_SeparatorActive]		= ImVec4(0.91f, 0.88f, 0.85f, 1.00f);
		Colors[ImGuiCol_ResizeGrip]				= ImVec4(0.93f, 0.92f, 0.89f, 0.25f);
		Colors[ImGuiCol_ResizeGripHovered]		= ImVec4(0.96f, 0.94f, 0.90f, 0.67f);
		Colors[ImGuiCol_ResizeGripActive]		= ImVec4(1.00f, 1.00f, 1.00f, 0.95f);
		Colors[ImGuiCol_Tab]					= ImVec4(0.39f, 0.38f, 0.36f, 0.86f);
		Colors[ImGuiCol_TabHovered]				= ImVec4(0.54f, 0.53f, 0.51f, 0.80f);
		Colors[ImGuiCol_TabActive]				= ImVec4(0.60f, 0.48f, 0.31f, 1.00f);
		Colors[ImGuiCol_TabUnfocusedActive]		= ImVec4(0.22f, 0.20f, 0.18f, 1.00f);
		Colors[ImGuiCol_DockingPreview]			= ImVec4(0.61f, 0.59f, 0.54f, 0.70f);
		Colors[ImGuiCol_DockingEmptyBg]			= ImVec4(0.18f, 0.18f, 0.18f, 0.00f);
		ImGui::GetStyle().WindowRounding = 0.0f;

		IO.Fonts->AddFontDefault();
		// ImFontConfig Config;
		// Config.MergeMode = true;
		// static const ImWchar IconRanges[] = { 0, 3000, 0 };
		// IO.Fonts->AddFontFromFileTTF("Resources\\Fonts\\ArialUnicode.ttf", 14.0f, &Config, IconRanges);

		Application& App = *Application::GetInstance();
		SDL_Window* Window = static_cast<SDL_Window*>(App.GetWindow().GetHandle());

		// Setup Platform/Renderer bindings
		ImGui_ImplSDL2_InitForOpenGL(Window, App.GetWindow().GetContext());
#ifndef ES_PLATFORM_APPLE
		ImGui_ImplOpenGL3_Init("#version 460");
#else
		ImGui_ImplOpenGL3_Init("#version 420");
#endif
	}

	void ImGuiLayer::OnDetach() {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
	}

	void ImGuiLayer::Begin() {
		SDL_Event Event;
		SDL_Window* Window = static_cast<SDL_Window*>(Application::GetInstance()->GetWindow().GetHandle());
		
		while (SDL_PollEvent(&Event))
			ImGui_ImplSDL2_ProcessEvent(&Event);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplSDL2_NewFrame(Window);
		ImGui::NewFrame();
	}

	void ImGuiLayer::End() {
		ImGuiIO& IO = ImGui::GetIO();
		Application& App = *Application::GetInstance();
		IO.DisplaySize = ImVec2((float)App.GetWindow().GetWidth(), (float)App.GetWindow().GetHeight());

		// Rendering
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		if (IO.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
			SDL_Window* backup_current_window = SDL_GL_GetCurrentWindow();
			SDL_GLContext backup_current_context = SDL_GL_GetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			SDL_GL_MakeCurrent(backup_current_window, backup_current_context);
		}

	}

	void ImGuiLayer::ShowApplicationDockspace(bool * Open) {
		static bool OptFullscreenPersistant = true;
		bool OptFullscreen = OptFullscreenPersistant;
		static ImGuiDockNodeFlags DockspaceFlags = ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar;

		// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
		// because it would be confusing to have two docking targets within each others.
		ImGuiWindowFlags WindowFlags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
		if (OptFullscreen) {
			ImGuiViewport* Viewport = ImGui::GetMainViewport();
			ImGui::SetNextWindowPos(Viewport->Pos);
			ImGui::SetNextWindowSize(Viewport->Size);
			ImGui::SetNextWindowViewport(Viewport->ID);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
			WindowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
			WindowFlags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
		}

		if (DockspaceFlags & ImGuiDockNodeFlags_PassthruCentralNode) {
			WindowFlags |= ImGuiWindowFlags_NoBackground;
			WindowFlags ^= ImGuiWindowFlags_MenuBar;
		}

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("DockSpace Demo", Open, WindowFlags);
		ImGui::PopStyleVar();

		if (OptFullscreen)
			ImGui::PopStyleVar(2);

		// DockSpace
		ImGuiIO& io = ImGui::GetIO();
		if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
			ImGuiID dockspace_id = ImGui::GetID("ApplicationDockSpace");
			ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), DockspaceFlags);
		}

		ImGui::End();
	}
	

	void ImGuiLayer::OnImGuiRender() {
		static bool Docking = true;
		ShowApplicationDockspace(&Docking);
		static bool show = true;
		ImGui::ShowDemoWindow(&show);
	}

}