#pragma once

#include "Core/Layer.h"

namespace ESource {

	class LayerStack {
	public:
		LayerStack();
		
		~LayerStack();

		void Clear();

		void PushLayer(Layer * NewLayer);

		void PopLayer(Layer * RemoveLayer);

		inline TArray<Layer *>::iterator begin() { return Layers.begin(); }
		inline TArray<Layer *>::iterator end() { return Layers.end(); }

	private:

		TArray<Layer *> Layers;
	};

}