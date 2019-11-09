#pragma once

#define MAX_INPUT_SCANCODE 512

namespace ESource {

	enum EButtonState : char {
		BS_Up = 0,
		BS_Down = 1,
		BS_Pressed = 2,
		BS_Released = 4,
		BS_Typed = 8,
	};

	struct InputScancodeState {
		int State;
		uint64_t FramePressed;
		int TypeRepeticions;
	};

	enum class EMouseButton {
		Mouse0 = 1,
		Mouse1,
		Mouse2,
		Mouse3,
		Mouse4,
		Mouse5
	};

	struct InputJoystickState {
		int State;
		uint64_t FramePressed;
	};
	
	// Taken from SDL Joystick for easy compatibility
	enum class EJoystickButton {
		Invalid = -1,
		RightPadDown,
		RightPadRight,
		RightPadLeft,
		RightPadUp,
		Back,
		Guide,
		Start,
		LeftStick,
		RightStick,
		LeftShoulder,
		RightShoulder,
		LeftPadUp,
		LeftPadDown,
		LeftPadLeft,
		LeftPadRight
	};

	enum class EJoystickAxis {
		Invalid = -1,
		LeftX,
		LeftY,
		RightX,
		RightY,
		TriggerLeft,
		TriggerRight
	};

	struct InputMouseButtonState {
		int State;
		uint64_t FramePressed;
		int Clicks;
	};

	/**
	 *  Taken from SDL Scancodes for easy compatibility
	 *  The values in this enumeration are based on the USB usage page standard:
	 *  https://www.usb.org/sites/default/files/documents/hut1_12v2.pdf
	 */
	enum class EScancode {
		Unknown = 0,
		
		A = 4,
		B = 5,
		C = 6,
		D = 7,
		E = 8,
		F = 9,
		G = 10,
		H = 11,
		I = 12,
		J = 13,
		K = 14,
		L = 15,
		M = 16,
		N = 17,
		O = 18,
		P = 19,
		Q = 20,
		R = 21,
		S = 22,
		T = 23,
		U = 24,
		V = 25,
		W = 26,
		X = 27,
		Y = 28,
		Z = 29,

		Num1 = 30,
		Num2 = 31,
		Num3 = 32,
		Num4 = 33,
		Num5 = 34,
		Num6 = 35,
		Num7 = 36,
		Num8 = 37,
		Num9 = 38,
		Num0 = 39,

		Return = 40,
		Escape = 41,
		Backspace = 42,
		Tab = 43,
		Space = 44,

		Minus = 45,
		Equals = 46,
		LeftBracket = 47,
		RightBracket = 48,
		Backslash = 49, /**< Located at the lower left of the return
						 *   key on ISO keyboards and at the right end
						 *   of the QWERTY row on ANSI keyboards.
						 *   Produces REVERSE SOLIDUS (backslash) and
						 *   VERTICAL LINE in a US layout, REVERSE
						 *   SOLIDUS and VERTICAL LINE in a UK Mac
						 *   layout, NUMBER SIGN and TILDE in a UK
						 *   Windows layout, DOLLAR SIGN and POUND SIGN
						 *   in a Swiss German layout, NUMBER SIGN and
						 *   APOSTROPHE in a German layout, GRAVE
						 *   ACCENT and POUND SIGN in a French Mac
						 *   layout, and ASTERISK and MICRO SIGN in a
						 *   French Windows layout.
						 */
		Semicolon = 51,
		Apostrophe = 52,
		GraveAccent = 53, /**< Located in the top left corner (on both ANSI
						   *   and ISO keyboards). Produces GRAVE ACCENT and
						   *   TILDE in a US Windows layout and in US and UK
						   *   Mac layouts on ANSI keyboards, GRAVE ACCENT
						   *   and NOT SIGN in a UK Windows layout, SECTION
						   *   SIGN and PLUS-MINUS SIGN in US and UK Mac
						   *   layouts on ISO keyboards, SECTION SIGN and
						   *   DEGREE SIGN in a Swiss German layout (Mac:
						   *   only on ISO keyboards), CIRCUMFLEX ACCENT and
						   *   DEGREE SIGN in a German layout (Mac: only on
						   *   ISO keyboards), SUPERSCRIPT TWO and TILDE in a
						   *   French Windows layout, COMMERCIAL AT and
						   *   NUMBER SIGN in a French Mac layout on ISO
						   *   keyboards, and LESS-THAN SIGN and GREATER-THAN
						   *   SIGN in a Swiss German, German, or French Mac
						   *   layout on ANSI keyboards.
						   */
		Comma = 54,
		Period = 55,
		Slash = 56,

		CapsLock = 57,

		F1 = 58,
		F2 = 59,
		F3 = 60,
		F4 = 61,
		F5 = 62,
		F6 = 63,
		F7 = 64,
		F8 = 65,
		F9 = 66,
		F10 = 67,
		F11 = 68,
		F12 = 69,

		Printscreen = 70,
		Scrolllock = 71,
		Pause = 72,
		Insert = 73, /**< insert on PC, help on some Mac keyboards (but does send code 73, not 117) */
		Home = 74,
		Pageup = 75,
		Delete = 76,
		End = 77,
		Pagedown = 78,
		Right = 79,
		Left = 80,
		Down = 81,
		Up = 82,

		NumLockClear = 83, /**< num lock on PC, clear on Mac keyboards
										 */
		KeypadDivide = 84,
		KeypadMultiply = 85,
		KeypadMinus = 86,
		KeypadPlus = 87,
		KeypadEnter = 88,
		Keypad1 = 89,
		Keypad2 = 90,
		Keypad3 = 91,
		Keypad4 = 92,
		Keypad5 = 93,
		Keypad6 = 94,
		Keypad7 = 95,
		Keypad8 = 96,
		Keypad9 = 97,
		Keypad0 = 98,
		KeypadPeriod = 99,

		NonUSBackslash = 100, /**< This is the additional key that ISO
							   *   keyboards have over ANSI ones,
							   *   located between left shift and Y.
							   *   Produces GRAVE ACCENT and TILDE in a
							   *   US or UK Mac layout, REVERSE SOLIDUS
							   *   (backslash) and VERTICAL LINE in a
							   *   US or UK Windows layout, and
							   *   LESS-THAN SIGN and GREATER-THAN SIGN
							   *   in a Swiss German, German, or French
							   *   layout. */
		Application = 101, /**< windows contextual menu, compose */
		Power = 102, /**< The USB document says this is a status flag,
					  *   not a physical key - but some Mac keyboards
					  *   do have a power key. */
		KeypadEquals = 103,
		F13 = 104,
		F14 = 105,
		F15 = 106,
		F16 = 107,
		F17 = 108,
		F18 = 109,
		F19 = 110,
		F20 = 111,
		F21 = 112,
		F22 = 113,
		F23 = 114,
		F24 = 115,
		Execute = 116,
		Help = 117,
		Menu = 118,
		Select = 119,
		Stop = 120,
		Again = 121,   /**< redo */
		Undo = 122,
		Cut = 123,
		Copy = 124,
		Paste = 125,
		Find = 126,
		Mute = 127,
		VolumeUp = 128,
		VolumeDown = 129,
		KeypadComma = 133,
		KeypadEqualsAS400 = 134,

		International1 = 135, /**< used on Asian keyboards, see footnotes in USB doc */
		International2 = 136,
		International3 = 137, /**< Yen */
		International4 = 138,
		International5 = 139,
		International6 = 140,
		International7 = 141,
		International8 = 142,
		International9 = 143,
		Lang1 = 144, /**< Hangul/English toggle */
		Lang2 = 145, /**< Hanja conversion */
		Lang3 = 146, /**< Katakana */
		Lang4 = 147, /**< Hiragana */
		Lang5 = 148, /**< Zenkaku/Hankaku */
		Lang6 = 149, /**< reserved */
		Lang7 = 150, /**< reserved */
		Lang8 = 151, /**< reserved */
		Lang9 = 152, /**< reserved */

		Alterase = 153, /**< erase-eaze */
		Sysreq = 154,
		Cancel = 155,
		Clear = 156,
		Prior = 157,
		Return2 = 158,
		Separator = 159,
		Out = 160,
		Oper = 161,
		ClearAgain = 162,
		Crsel = 163,
		Exsel = 164,

		Keypad00 = 176,
		Keypad000 = 177,
		ThousandsSeparator = 178,
		DecimalSeparator = 179,
		CurrencyUnit = 180,
		CurrencySubunit = 181,

		LeftCtrl = 224,
		LeftShift = 225,
		LeftAlt = 226, /**< alt, option */
		LeftGui = 227, /**< windows, command (apple), meta */
		RightCtrl = 228,
		RightShift = 229,
		RightAlt = 230, /**< alt gr, option */
		RightGui = 231, /**< windows, command (apple), meta */

		Mode = 257,    /**< I'm not sure if this is really not covered
						*   by any of the above, but since there's a
						*   special KMOD_MODE for it I'm adding it here
						*/

		AudioNext = 258,
		AudioPrev = 259,
		AudioStop = 260,
		AudioPlay = 261,
		AudioMute = 262,
		MediaSelect = 263,
		WWW = 264,
		Mail = 265,
		Calculator = 266,
		Computer = 267,
		AC_Search = 268,
		AC_Home = 269,
		AC_Back = 270,
		AC_Forward = 271,
		AC_Stop = 272,
		AC_Refresh = 273,
		AC_Bookmarks = 274,

		BrightnessDown = 275,
		BrightnessUp = 276,
		DisplaySwitch = 277, /**< display mirroring/dual display switch, video mode switch */
		KBDILLUMTOGGLE = 278,
		KBDILLUMDOWN = 279,
		KBDILLUMUP = 280,
		Eject = 281,
		Sleep = 282,

		AudioRewind = 285,
		AudioFastForward = 286
	};

}