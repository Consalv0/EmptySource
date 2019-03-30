
#include "../include/Bitmap.h"
#include <memory>

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

template <typename T>
T & Bitmap<T>::operator()(int x, int y) {
	return Data[y * Width + x];
}

template <typename T>
const T & Bitmap<T>::operator()(int x, int y) const {
	return Data[y * Width + x];
}

template class Bitmap<UCharRed>;
template class Bitmap<UCharRGB>;
template class Bitmap<UCharRGBA>;
template class Bitmap<FloatRed>;
template class Bitmap<FloatRGB>;
template class Bitmap<FloatRGBA>;
