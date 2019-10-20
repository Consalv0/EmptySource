#pragma once

namespace ESource {

	class PixelMap {
	public:
		PixelMap();

		PixelMap(int Width, int Height, int Depth, EPixelFormat PixelFormat);

		PixelMap(int Width, int Height, int Depth, EPixelFormat PixelFormat, void *& Data);
		
		PixelMap(const PixelMap &Other);

		void SetData(int Width, int Height, int Depth, EPixelFormat PixelFormat, void *& InData);

		//* Width in pixels. 
		inline uint32_t GetWidth() const { return Width; };

		//* Height in pixels.
		inline uint32_t GetHeight() const { return Height; };

		//* Depth in pixels.
		inline uint32_t GetDepth() const { return Depth; };

		inline IntVector3 GetSize() const { return { (int)Width, (int)Height, (int)Depth }; }

		inline bool IsEmpty() const { return Data == NULL; };

		size_t GetMemorySize() const;

		inline EPixelFormat GetColorFormat() const { return PixelFormat; };

		PixelMap & operator=(const PixelMap & Other);
		
		const void * PointerToValue() const;

		~PixelMap();

	private:
		friend class PixelMapUtility;

		void * Data;
		EPixelFormat PixelFormat;
		uint32_t Width, Height, Depth;
	};

	class PixelMapUtility {
	public:
		static void CreateData(int Width, int Height, int Depth, EPixelFormat PixelFormat, void *& Data);

		static void CreateData(int Width, int Height, int Depth, EPixelFormat PixelFormat, void *& Target, void * Data);

		//* Flips the pixels vertically
		static void FlipVertically(PixelMap & Map);

		static unsigned char * GetCharPixelAt(PixelMap & Map, const uint32_t & X, const uint32_t & Y, const uint32_t & Z);

		static float * GetFloatPixelAt(PixelMap & Map, const uint32_t & X, const uint32_t & Y, const uint32_t & Z);

		static unsigned char * GetCharPixelAt(PixelMap & Map, const size_t & Index);

		static float * GetFloatPixelAt(PixelMap & Map, const size_t & Index);

		static void PerPixelOperator(PixelMap & Map, std::function<void(float *, const unsigned char & Channels)> const& Function);

		static void PerPixelOperator(PixelMap & Map, std::function<void(unsigned char *, const unsigned char & Channels)> const& Function);

		static bool FormatIsFloat(EPixelFormat Format);

		static bool FormatIsShort(EPixelFormat Format);

	private:
		template<typename T>
		static void _FlipVertically(uint32_t Width, uint32_t Height, uint32_t Depth, void *& Data);
	};

	template<typename T>
	inline void PixelMapUtility::_FlipVertically(uint32_t Width, uint32_t Height, uint32_t Depth, void *& InData) {
		T* Data = (T*)InData;
		T* TempRow = (T*)malloc(Width * sizeof(T));
		for (uint32_t DepthIndex = 0; DepthIndex < Depth; DepthIndex++) {
			uint32_t DepthOffset = Width * Height * DepthIndex;
			// Iterate only half the buffer to get a full flip
			uint32_t Rows = Height / 2;

			for (uint32_t RowIndex = 0; RowIndex < Rows; RowIndex++) {
				memcpy(TempRow, Data + DepthOffset + RowIndex * Width, Width * sizeof(T));
				memcpy(Data + DepthOffset + RowIndex * Width, Data + DepthOffset + (Height - RowIndex - 1) * Width, Width * sizeof(T));
				memcpy(Data + DepthOffset + (Height - RowIndex - 1) * Width, TempRow, Width * sizeof(T));
			}
		}
		free(TempRow);
	}

}