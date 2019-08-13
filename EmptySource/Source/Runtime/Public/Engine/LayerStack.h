#pragma once

#include "Engine/Layer.h"

namespace EmptySource {

	class LayerStack {
	public:
		LayerStack();
		
		~LayerStack();

		void Clear();

		void PushLayer(Layer * NewLayer);

		void PushOverlay(Layer * Overlay);
		
		void PopLayer(Layer * PopingLayer);
		
		void PopOverlay(Layer * PopingOverlay);

		TArray<Layer *>::iterator begin() { return Layers.begin(); }
		TArray<Layer *>::iterator end() { return Layers.end(); }

	private:

		TArray<Layer *> Layers;
		TArray<Layer *>::iterator LayerInsert;
	};

}