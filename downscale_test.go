package tcgvision

import (
	"image"
	"testing"
)

func TestDownscale(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 3000, 2000))
	got := Downscale(src, 1500)
	if b := got.Bounds(); b.Dx() != 1500 || b.Dy() != 1000 {
		t.Fatalf("downscaled to %dx%d", b.Dx(), b.Dy())
	}

	same := Downscale(src, 4000)
	if same != image.Image(src) {
		t.Error("image within bounds must be returned unchanged")
	}
}
