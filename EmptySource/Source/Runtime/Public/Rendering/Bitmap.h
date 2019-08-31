#pragma once

namespace EmptySource {

	template <typename T>
	class Bitmap {

	public:
		Bitmap();
		Bitmap(int Width, int Height);
		Bitmap(const Bitmap<T> &Other);

		//* Width in pixels. 
		int GetWidth() const;

		//* Height in pixels.
		int GetHeight() const;

		inline IntVector2 GetSize() const { return { GetWidth(), GetHeight() }; }

		//* Flips the pixels vertically
		void FlipVertically();

		Bitmap<T> & operator=(const Bitmap<T> & Other);
		
		T const& operator[](int i) const;
		     T & operator[](int i);
		
		const T & operator()(int x, int y) const;
		      T & operator()(int x, int y);
		
		const T* PointerToValue() const;

		template<typename U>
		void ChangeType(Bitmap<U> & Result, std::function<U(T)> const& Function) const {
			Result = Bitmap<U>(Width, Height);
			for (int y = 0; y < Height; ++y) {
				for (int x = 0; x < Width; ++x) {
					Result(x, y) = Function((*this)(x, y));
				}
			}
		}

		void PerPixelOperator(std::function<void(T&)> const& Function);

		~Bitmap();

	private:
		T * Data;
		int Width, Height;
	};

}