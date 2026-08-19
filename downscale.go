package tcgvision

import "image"

// Downscale returns img resized so its longer side is at most maxSide,
// preserving aspect ratio (bilinear). Returns img unchanged when it already
// fits. Use it to bound the pipeline's memory and latency on large photos:
// detection and embedding quality are unaffected well above the detector's
// input size.
func Downscale(img image.Image, maxSide int) image.Image {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= maxSide && h <= maxSide {
		return img
	}

	scale := float32(maxSide) / float32(w)
	if s := float32(maxSide) / float32(h); s < scale {
		scale = s
	}
	dw := int(float32(w)*scale + 0.5)
	dh := int(float32(h)*scale + 0.5)

	small := resizeBilinear(fromImage(img), dw, dh)
	out := image.NewRGBA(image.Rect(0, 0, dw, dh))
	i := 0
	for y := range dh {
		for x := range dw {
			o := y*out.Stride + x*4
			out.Pix[o] = uint8(small.r[i] + 0.5)
			out.Pix[o+1] = uint8(small.g[i] + 0.5)
			out.Pix[o+2] = uint8(small.b[i] + 0.5)
			out.Pix[o+3] = 255
			i++
		}
	}

	return out
}
