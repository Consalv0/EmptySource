#pragma once

#include "Math/MathUtility.h"
#include "Math/Vector2.h"
#include "Math/IntVector2.h"
#include "Math/Box2D.h"

namespace ESource {

	template <typename T>
	class TexturePacking {
	public:
		struct ReturnElement {
			bool bValid;
			Box2D BBox;
			const T * Element;
		};

	private:
		struct Node {
			Node* Smaller;
			Node* Bigger;
			Box2D BBox;
			const T * Element;

			Node* Insert(const T & Element);
			ReturnElement ReturnAsElement() const;
			~Node();
		};

		Node * PNode;

	public:
		~TexturePacking();

		void CreateTexture(const IntVector2& Dimensions);
		TexturePacking<T>::ReturnElement Insert(const T & Element);
	};

}