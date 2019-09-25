#pragma once

namespace EmptySource {

	class PixelMap {
	public:
		PixelMap();

		PixelMap(int Width, int Height, int Depth, EColorFormat PixelFormat);
		
		PixelMap(const PixelMap &Other);

		//* Width in pixels. 
		inline unsigned int GetWidth() const { return Width; };

		//* Height in pixels.
		inline unsigned int GetHeight() const { return Height; };

		//* Depth in pixels.
		inline unsigned int GetDepth() const { return Depth; };

		inline IntVector3 GetDimensions() const { return { (int)Width, (int)Height, (int)Depth }; }

		inline bool IsEmpty() const { return Data == NULL; };

		size_t GetMemorySize() const;

		inline EColorFormat GetColorFormat() const { return PixelFormat; };

		PixelMap & operator=(const PixelMap & Other);
		
		const void * PointerToValue() const;

		~PixelMap();

	private:
		friend class PixelMapUtility;

		void * Data;
		EColorFormat PixelFormat;
		unsigned int Width, Height, Depth;
	};

	class PixelMapUtility {
	public:
		static void CreateData(int Width, int Height, int Depth, EColorFormat PixelFormat, void *& Data);

		static unsigned int PixelSize(const EColorFormat & Format);

		static unsigned int PixelSize(const PixelMap & Map);

		static unsigned int PixelChannels(const EColorFormat & Format);

		//* Flips the pixels vertically
		static void FlipVertically(PixelMap & Map);

		static unsigned char * GetCharPixelAt(PixelMap & Map, const unsigned int & X, const unsigned int & Y, const unsigned int & Z);

		static float * GetFloatPixelAt(PixelMap & Map, const unsigned int & X, const unsigned int & Y, const unsigned int & Z);

		static unsigned char * GetCharPixelAt(PixelMap & Map, const size_t & Index);

		static float * GetFloatPixelAt(PixelMap & Map, const size_t & Index);

		static void PerPixelOperator(PixelMap & Map, std::function<void(float *, const unsigned char & Channels)> const& Function);

		static void PerPixelOperator(PixelMap & Map, std::function<void(unsigned char *, const unsigned char & Channels)> const& Function);

		static bool ColorFormatIsFloat(EColorFormat Format);

	private:
		template<typename T>
		static void _FlipVertically(unsigned int Width, unsigned int Height, unsigned int Depth, void *& Data);
	};

	template<typename T>
	inline void PixelMapUtility::_FlipVertically(unsigned int Width, unsigned int Height, unsigned int Depth, void *& InData) {
		T* Data = (T*)InData;
		T* TempRow = (T*)malloc(Width * sizeof(T));
		for (unsigned int DepthIndex = 0; DepthIndex < Depth; DepthIndex++) {
			unsigned int DepthOffset = Width * Height * DepthIndex;
			// Iterate only half the buffer to get a full flip
			unsigned int Rows = Height / 2;

			for (unsigned int RowIndex = 0; RowIndex < Rows; RowIndex++) {
				memcpy(TempRow, Data + DepthOffset + RowIndex * Width, Width * sizeof(T));
				memcpy(Data + DepthOffset + RowIndex * Width, Data + DepthOffset + (Height - RowIndex - 1) * Width, Width * sizeof(T));
				memcpy(Data + DepthOffset + (Height - RowIndex - 1) * Width, TempRow, Width * sizeof(T));
			}
		}
		free(TempRow);
	}

}