#pragma once

#include "Graphics/Graphics.h"

#include <functional>

namespace EmptySource {

	template <typename T>
	class Bitmap {

	public:
		Bitmap();
		Bitmap(int width, int height);
		Bitmap(const Bitmap<T> &orig);

		// --- Width in pixels. 
		int GetWidth() const;

		// --- Height in pixels.
		int GetHeight() const;

		void FlipVertically();

		Bitmap<T> & operator=(const Bitmap<T> & Other);
		T & operator[](int i);
		T const& operator[](int i) const;
		const T* PointerToValue() const;
		T & operator()(int x, int y);
		const T & operator()(int x, int y) const;

		template<typename U>
		void ChangeType(Bitmap<U> & Result, std::function<U(T)> const& Function) const {
			Result = Bitmap<U>(Width, Height);
			for (int y = 0; y < Height; ++y) {
				for (int x = 0; x < Width; ++x) {
					Result(x, y) = Function((*this)(x, y));
				}
			}
		}

		~Bitmap();

	private:
		T * Data;
		int Width, Height;
	};

}