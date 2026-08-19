package tcgvision

import "math"

// float32ToF16 converts a float32 to IEEE 754 binary16 (round-to-nearest-even,
// overflow to inf, subnormals preserved). Matches numpy's float16 cast for the
// value range embeddings live in.
func float32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23&0xff) - 127 + 15
	mant := bits & 0x7fffff

	switch {
	case exp >= 31: // overflow or inf/nan
		if bits&0x7fffffff > 0x7f800000 { // nan
			return sign | 0x7e00
		}

		return sign | 0x7c00
	case exp <= 0: // subnormal or zero
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint32(1) << (shift - 1)
		rounded := (mant + half - 1 + (mant>>shift)&1) >> shift

		return sign | uint16(rounded)
	default:
		rounded := mant + 0xfff + (mant>>13)&1
		if rounded&0x800000 != 0 { // mantissa overflowed into exponent
			rounded = 0
			exp++
			if exp >= 31 {
				return sign | 0x7c00
			}
		}

		return sign | uint16(exp)<<10 | uint16(rounded>>13)
	}
}

// f16ToFloat32 converts IEEE 754 binary16 to float32.
func f16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h >> 10 & 0x1f)
	mant := uint32(h & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal: normalize
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3ff
		exp++

		return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
	case 31:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
	}
}
