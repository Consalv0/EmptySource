
#include "CoreMinimal.h"
#include "Engine/LayerStack.h"

namespace EmptySource {

	LayerStack::LayerStack() {
		LayerInsert = Layers.begin();
	}

	LayerStack::~LayerStack() {
		Clear();
	}

	void LayerStack::Clear() {
		for (Layer * ItLayer : Layers)
			delete ItLayer;
	}

	void LayerStack::PushLayer(Layer * NewLayer) {
		LayerInsert = Layers.emplace(LayerInsert, NewLayer);
	}

	void LayerStack::PushOverlay(Layer * NewOverlay) {
		Layers.emplace_back(NewOverlay);
	}

	void LayerStack::PopLayer(Layer * PopingLayer) {
		auto LayerIt = std::find(Layers.begin(), Layers.end(), PopingLayer);
		if (LayerIt != Layers.end()) {
			Layers.erase(LayerIt);
			LayerInsert--;
		}
	}

	void LayerStack::PopOverlay(Layer * PopingOverlay) {
		auto LayerIt = std::find(Layers.begin(), Layers.end(), PopingOverlay);
		if (LayerIt != Layers.end()) {
			Layers.erase(LayerIt);
		}
	}

}