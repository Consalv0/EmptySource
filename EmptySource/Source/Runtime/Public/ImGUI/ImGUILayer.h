#pragma once

#include "Core/Layer.h"

#include "Core/Application.h"
#include "Events/WindowEvent.h"
#include "Events/InputEvent.h"

namespace ESource {

	class ImGuiLayer : public Layer {
	public:
		ImGuiLayer();
		~ImGuiLayer() = default;

		virtual void OnAwake() override;
		virtual void OnDetach() override;
		virtual void OnImGuiRender() override;

		void Begin();
		void End();

	private:
		float LayerTime = 0.F;

		void ShowApplicationDockspace(bool * Open);
	};

}