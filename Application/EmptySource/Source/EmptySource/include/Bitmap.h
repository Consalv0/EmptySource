#pragma once

template <typename T>
class Bitmap {

public:
	Bitmap();
	Bitmap(int width, int height);
	Bitmap(const Bitmap<T> &orig);

	// Width in pixels. 
	int GetWidth() const;

	// Height in pixels.
	int GetHeight() const;

	Bitmap<T> & operator=(const Bitmap<T> & Other);
	T & operator[](int i);
	T const& operator[](int i) const;
	const T* PointerToValue() const;
	T & operator()(int x, int y);
	const T & operator()(int x, int y) const;

	~Bitmap();

private:
	T * Data;
	int Width, Height;
};
