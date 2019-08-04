
#include "../include/Bitmap.h"
#include <memory>

namespace EmptySource {

	template <typename T>
	Bitmap<T>::Bitmap() : Data(NULL), Width(0), Height(0) { }

	template <typename T>
	Bitmap<T>::Bitmap(int width, int height) : Width(width), Height(height) {
		Data = new T[Width * Height];
	}

	template <typename T>
	Bitmap<T>::Bitmap(const Bitmap<T> &Other) : Width(Other.Width), Height(Other.Height) {
		Data = new T[Width * Height];
		memcpy(Data, Other.Data, Width * Height * sizeof(T));
	}

	template <typename T>
	Bitmap<T>::~Bitmap() {
		delete[] Data;
	}

	template <typename T>
	Bitmap<T> & Bitmap<T>::operator=(const Bitmap<T> &Other) {
		delete[] Data;
		Width = Other.Width, Height = Other.Height;
		Data = new T[Width * Height];
		memcpy(Data, Other.Data, Width * Height * sizeof(T));
		return *this;
	}

	template<typename T>
	T & Bitmap<T>::operator[](int i) {
		return Data[i];
	}

	template<typename T>
	T const & Bitmap<T>::operator[](int i) const {
		return Data[i];
	}

	template<typename T>
	const T * Bitmap<T>::PointerToValue() const {
		return Data;
	}

	template <typename T>
	int Bitmap<T>::GetWidth() const {
		return Width;
	}

	template <typename T>
	int Bitmap<T>::GetHeight() const {
		return Height;
	}

	template<typename T>
	void Bitmap<T>::FlipVertically() {
		unsigned Rows = Height / 2; // Iterate only half the buffer to get a full flip
		T* TempRow = (T*)malloc(Width * sizeof(T));

		for (unsigned RowIndex = 0; RowIndex < Rows; RowIndex++) {
			memcpy(TempRow, Data + RowIndex * Width, Width * sizeof(T));
			memcpy(Data + RowIndex * Width, Data + (Height - RowIndex - 1) * Width, Width * sizeof(T));
			memcpy(Data + (Height - RowIndex - 1) * Width, TempRow, Width * sizeof(T));
		}

		free(TempRow);
	}

	template <typename T>
	T & Bitmap<T>::operator()(int x, int y) {
		return Data[y * Width + x];
	}

	template <typename T>
	const T & Bitmap<T>::operator()(int x, int y) const {
		return Data[y * Width + x];
	}

	template class Bitmap<UCharRed>;
	template class Bitmap<UCharRG>;
	template class Bitmap<UCharRGB>;
	template class Bitmap<UCharRGBA>;
	template class Bitmap<FloatRed>;
	template class Bitmap<FloatRG>;
	template class Bitmap<FloatRGB>;
	template class Bitmap<FloatRGBA>;

}