
#include "CoreMinimal.h"
#include "Core/LayerStack.h"

namespace ESource {

	LayerStack::LayerStack() {
	}

	LayerStack::~LayerStack() {
		Clear();
	}

	void LayerStack::Clear() {
		for (Layer * ItLayer : Layers)
			delete ItLayer;
	}

	void LayerStack::PushLayer(Layer * NewLayer) {
		auto LayerIt = begin();
		for (; LayerIt != end(); ++LayerIt) {
			if (*(*LayerIt) >= *NewLayer)
				break;
		}
		Layers.emplace(LayerIt, NewLayer);
	}

	void LayerStack::PopLayer(Layer * RemoveLayer) {
		auto LayerIt = std::find(begin(), end(), RemoveLayer);
		if (LayerIt != Layers.end()) {
			Layers.erase(LayerIt);
		}
	}

}