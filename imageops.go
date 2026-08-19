package tcgvision

import (
	"image"
)

// rgbImage is a planar float32 RGB image with values in [0, 255].
// All internal image math works on this representation.
type rgbImage struct {
	w, h    int
	r, g, b []float32
}

func newRGBImage(w, h int) *rgbImage {
	n := w * h

	return &rgbImage{w: w, h: h, r: make([]float32, n), g: make([]float32, n), b: make([]float32, n)}
}

// fromImage converts any image.Image into an rgbImage. The fast path handles
// *image.YCbCr (what image/jpeg produces) and *image.RGBA/NRGBA.
func fromImage(img image.Image) *rgbImage {
	bounds := img.Bounds()
	out := newRGBImage(bounds.Dx(), bounds.Dy())

	i := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			out.r[i] = float32(r >> 8)
			out.g[i] = float32(g >> 8)
			out.b[i] = float32(b >> 8)
			i++
		}
	}

	return out
}

// bilinearAt samples one channel at continuous coordinates with clamping,
// matching OpenCV INTER_LINEAR (half-pixel centers are applied by callers).
func bilinearAt(p []float32, w, h int, x, y float32) float32 {
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	fx := float32(w - 1)
	if x > fx {
		x = fx
	}
	fy := float32(h - 1)
	if y > fy {
		y = fy
	}

	x0 := int(x)
	y0 := int(y)
	x1 := x0 + 1
	if x1 > w-1 {
		x1 = w - 1
	}
	y1 := y0 + 1
	if y1 > h-1 {
		y1 = h - 1
	}
	dx := x - float32(x0)
	dy := y - float32(y0)

	top := p[y0*w+x0]*(1-dx) + p[y0*w+x1]*dx
	bot := p[y1*w+x0]*(1-dx) + p[y1*w+x1]*dx

	return top*(1-dy) + bot*dy
}

// resizeBilinear resizes src to (dw, dh) using half-pixel-center bilinear
// interpolation (the OpenCV INTER_LINEAR convention).
func resizeBilinear(src *rgbImage, dw, dh int) *rgbImage {
	dst := newRGBImage(dw, dh)
	sx := float32(src.w) / float32(dw)
	sy := float32(src.h) / float32(dh)

	i := 0
	for y := range dh {
		fy := (float32(y)+0.5)*sy - 0.5
		for x := range dw {
			fx := (float32(x)+0.5)*sx - 0.5
			dst.r[i] = bilinearAt(src.r, src.w, src.h, fx, fy)
			dst.g[i] = bilinearAt(src.g, src.w, src.h, fx, fy)
			dst.b[i] = bilinearAt(src.b, src.w, src.h, fx, fy)
			i++
		}
	}

	return dst
}

// crop returns the sub-image [x0,x1)x[y0,y1) clamped to the source bounds.
func (m *rgbImage) crop(x0, y0, x1, y1 int) *rgbImage {
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > m.w {
		x1 = m.w
	}
	if y1 > m.h {
		y1 = m.h
	}

	dst := newRGBImage(x1-x0, y1-y0)
	i := 0
	for y := y0; y < y1; y++ {
		base := y * m.w
		for x := x0; x < x1; x++ {
			dst.r[i] = m.r[base+x]
			dst.g[i] = m.g[base+x]
			dst.b[i] = m.b[base+x]
			i++
		}
	}

	return dst
}
