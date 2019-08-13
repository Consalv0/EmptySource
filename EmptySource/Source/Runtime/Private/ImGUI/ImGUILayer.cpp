
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

		// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
		// ImGuiStyle& style = ImGui::GetStyle();
		// if (IO.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
		// 	style.WindowRounding = 0.0f;
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