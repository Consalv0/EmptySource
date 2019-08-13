
#include "CoreMinimal.h"
#include <imgui.h>
#include <examples/imgui_impl_sdl.h>
#include <examples/imgui_impl_opengl3.h>

#include "ImGUI/ImGUILayer.h"

// TEMPORARY
#include <SDL.h>
#include <GLAD/glad.h>

namespace EmptySource {

	ImGUILayer::ImGUILayer()
		: Layer(L"ImGUILayer") {
	}

	void ImGUILayer::OnAwake() {
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& IO = ImGui::GetIO(); (void)IO;
		IO.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
		// IO.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;     // Enable Gamepad Controls
		// IO.ConfigFlags |= ImGuiConfigFlags_DockingEnable;        // Enable Docking
		// IO.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;      // Enable Multi-Viewport / Platform Windows
		// IO.ConfigFlags |= ImGuiConfigFlags_ViewportsNoTaskBarIcons;
		// IO.ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge;

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();
		ImVec4* Colors = ImGui::GetStyle().Colors;
		Colors[ImGuiCol_Text]                   = ImVec4(0.99f, 0.96f, 0.93f, 1.00f);
		Colors[ImGuiCol_FrameBg]                = ImVec4(0.22f, 0.22f, 0.22f, 0.54f);
		Colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.56f, 0.56f, 0.56f, 0.40f);
		Colors[ImGuiCol_FrameBgActive]          = ImVec4(0.65f, 0.65f, 0.65f, 0.67f);
		Colors[ImGuiCol_TitleBgActive]          = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
		Colors[ImGuiCol_CheckMark]              = ImVec4(0.96f, 0.91f, 0.88f, 1.00f);
		Colors[ImGuiCol_SliderGrab]             = ImVec4(0.96f, 0.91f, 0.88f, 1.00f);
		Colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.97f, 0.95f, 0.93f, 1.00f);
		Colors[ImGuiCol_Button]                 = ImVec4(0.96f, 0.91f, 0.88f, 0.13f);
		Colors[ImGuiCol_ButtonHovered]          = ImVec4(0.96f, 0.91f, 0.88f, 0.32f);
		Colors[ImGuiCol_ButtonActive]           = ImVec4(0.96f, 0.91f, 0.98f, 0.40f);
		Colors[ImGuiCol_Header]                 = ImVec4(0.39f, 0.39f, 0.39f, 0.31f);
		Colors[ImGuiCol_HeaderHovered]          = ImVec4(0.47f, 0.47f, 0.47f, 0.80f);
		Colors[ImGuiCol_HeaderActive]           = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
		Colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.82f, 0.82f, 0.82f, 0.78f);
		Colors[ImGuiCol_SeparatorActive]        = ImVec4(0.91f, 0.88f, 0.85f, 1.00f);
		Colors[ImGuiCol_ResizeGrip]             = ImVec4(0.93f, 0.92f, 0.89f, 0.25f);
		Colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.96f, 0.94f, 0.90f, 0.67f);
		Colors[ImGuiCol_ResizeGripActive]       = ImVec4(1.00f, 1.00f, 1.00f, 0.95f);
		Colors[ImGuiCol_Tab]                    = ImVec4(0.39f, 0.38f, 0.36f, 0.86f);
		Colors[ImGuiCol_TabHovered]             = ImVec4(0.54f, 0.53f, 0.51f, 0.80f);
		Colors[ImGuiCol_TabActive]              = ImVec4(0.60f, 0.48f, 0.31f, 1.00f);
		Colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.22f, 0.20f, 0.18f, 1.00f);
		Colors[ImGuiCol_DockingPreview]         = ImVec4(0.61f, 0.59f, 0.54f, 0.70f);
		ImGui::GetStyle().WindowRounding = 0.0f;

		// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
		// ImGuiStyle& style = ImGui::GetStyle();
		// if (IO.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
		// 	style.Colors[ImGuiCol_WindowBg].w = 1.0f;
		// }

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

	void ImGUILayer::OnDetach() {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
	}

	void ImGUILayer::Begin() {
		SDL_Window* Window = static_cast<SDL_Window*>(Application::GetInstance()->GetWindow().GetHandle());
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplSDL2_NewFrame(Window);
		ImGui::NewFrame();

		SDL_Event Event;
		while (SDL_PollEvent(&Event)) {
			ImGui_ImplSDL2_ProcessEvent(&Event);
		}
	}

	void ImGUILayer::End() {
		ImGuiIO& IO = ImGui::GetIO();
		Application& App = *Application::GetInstance();
		IO.DisplaySize = ImVec2((float)App.GetWindow().GetWidth(), (float)App.GetWindow().GetHeight());

		// Rendering
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// if (IO.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
		// 	SDL_Window* backup_current_context = static_cast<SDL_Window*>(Application::GetInstance()->GetWindow().GetHandle());
		// 	ImGui::UpdatePlatformWindows();
		// 	ImGui::RenderPlatformWindowsDefault();
		// 	SDL_Current(backup_current_context);
		// }
	}

	void ImGUILayer::OnImGUIRender() {
		static bool show = true;
		ImGui::ShowDemoWindow(&show);
	}

}