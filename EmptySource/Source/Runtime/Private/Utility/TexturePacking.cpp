
#include "CoreMinimal.h"

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/PixelMap.h"
#include "Utility/TexturePacking.h"

namespace ESource {

	template<typename T>
	typename TexturePacking<T>::Node * TexturePacking<T>::Node::Insert(const T & Element) {
		// --- We're not in leaf
		if (Smaller != NULL) {
			// --- Try inserting into first child
			Node * NewNode = Smaller->Insert(Element);
			if (NewNode != NULL) return NewNode;

			// --- No room, insert into second
			return Bigger->Insert(Element);
		}
		else {
			// --- If There's already a element here, return;
			if (this->Element != NULL) return NULL;

			// --- If We're too small, return
			if ((int)Element.GetWidth() > (int)BBox.GetWidth() ||
				(int)Element.GetHeight() > (int)BBox.GetHeight())
				return NULL;

			// --- Removed: resolved the stuck subdivision
			// --- If We're just right, accept
			// if (Element.SDFResterized.GetWidth() == (int)BBox.GetWidth() &&
			//  	Element.SDFResterized.GetHeight() == (int)BBox.GetHeight())
			//     return this;

			// --- Otherwise, gotta split this node and create some kids
			Smaller = new Node();
			Bigger = new Node();

			int Width = Element.GetWidth();
			int Height = Element.GetHeight();

			int DeltaWidth = (int)BBox.GetWidth() - Width;
			int DeltaHeight = (int)BBox.GetHeight() - Height;

			// --- Decide which way to split
			if (DeltaWidth > DeltaHeight) {
				Smaller->BBox = { BBox.MinX,         BBox.MinY + Height, BBox.MinX + Width, BBox.MaxY };
				Bigger->BBox = { BBox.MinX + Width, BBox.MinY,          BBox.MaxX,         BBox.MaxY };
			}
			else {
				Smaller->BBox = { BBox.MinX + Width, BBox.MinY,          BBox.MaxX, BBox.MinY + Height };
				Bigger->BBox = { BBox.MinX,         BBox.MinY + Height, BBox.MaxX, BBox.MaxY };
			}

			this->Element = &Element;
			return this;
		}
	}

	template<typename T>
	typename TexturePacking<T>::ReturnElement TexturePacking<T>::Node::ReturnAsElement() const {
		return { true, BBox, Element };
	}

	template<typename T>
	TexturePacking<T>::Node::~Node() {
		if (Smaller != NULL)
			delete Smaller;
		if (Bigger != NULL)
			delete Bigger;
	}

	template<typename T>
	TexturePacking<T>::~TexturePacking() {
		if (PNode)
			delete PNode;
	}

	template<typename T>
	inline void TexturePacking<T>::CreateTexture(const IntVector2& Dimensions) {
		PNode = new Node();
		PNode->BBox = { 0.F, 0.F, (float)Dimensions.X, (float)Dimensions.Y };
	}

	template<typename T>
	typename TexturePacking<T>::ReturnElement TexturePacking<T>::Insert(const T & Element) {
		if (PNode == NULL)
			return { false, BoundingBox2D(), (T*)(NULL) };

		Node * ReturnNode = PNode->Insert(Element);
		if (ReturnNode)
			return ReturnNode->ReturnAsElement();

		return { false, BoundingBox2D(), (T*)(NULL) };
	}

	template class TexturePacking<PixelMap>;

}