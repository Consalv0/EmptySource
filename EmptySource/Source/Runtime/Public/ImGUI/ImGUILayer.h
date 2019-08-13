#pragma once

#include "Engine/Layer.h"

#include "Engine/Application.h"
#include "Events/WindowEvent.h"
#include "Events/InputEvent.h"

namespace EmptySource {

	class ImGUILayer : public Layer {
	public:
		ImGUILayer();
		~ImGUILayer() = default;

		virtual void OnAwake() override;
		virtual void OnDetach() override;
		virtual void OnImGUIRender() override;

		void Begin();
		void End();

	private:
		float LayerTime = 0.F;
	};

}