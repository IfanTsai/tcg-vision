package tcgvision

import (
	"errors"
	"image"
	"image/color"
	"math"
	"os"
	"testing"
)

func TestGundamProfile(t *testing.T) {
	prof := Gundam()

	// Rectified to the physical 63x88mm aspect.
	if got, want := float64(prof.CardH)/float64(prof.CardW), 88.0/63.0; math.Abs(got-want) > 0.005 {
		t.Errorf("card aspect %.4f, want %.4f (63x88mm)", got, want)
	}

	if prof.ArtX < 0 || prof.ArtY < 0 || prof.ArtX+prof.ArtW > 1 || prof.ArtY+prof.ArtH > 1 {
		t.Errorf("artwork window %+v leaves the card", prof)
	}

	// An official render (600x838) takes the whole-image fallback.
	if !flatAspectOK(600, 838, prof) {
		t.Error("official 600x838 render should take the flat fallback")
	}

	if prof.DetectSize != YuGiOh().DetectSize {
		t.Error("detection parameters should follow the shared detector")
	}
}

// TestWithProfileSharesModels checks that a derived pipeline reuses the loaded
// sessions, embeds with its own artwork window, and is closed together with
// its parent. Needs the embedder model (testdata/private/models).
func TestWithProfileSharesModels(t *testing.T) {
	const embedderPath = "testdata/private/models/embedder.onnx"
	if _, err := os.Stat(embedderPath); err != nil {
		t.Skipf("embedder model not available: %v", err)
	}

	pipe, err := New(Config{ORTLibPath: ortLibPath(t), EmbedderPath: embedderPath})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	gundam := pipe.WithProfile(Gundam())
	if gundam.models != pipe.models {
		t.Fatal("WithProfile loaded its own models")
	}

	if pipe.Profile() != YuGiOh() || gundam.Profile() != Gundam() {
		t.Fatal("profiles leaked between pipelines")
	}

	card := splitCard(480, 670)
	a, err := pipe.EmbedReference(card)
	if err != nil {
		t.Fatalf("EmbedReference (yugioh): %v", err)
	}

	b, err := gundam.EmbedReference(card)
	if err != nil {
		t.Fatalf("EmbedReference (gundam): %v", err)
	}

	if dot(a, b) > 0.999 {
		t.Error("both profiles embedded the same window")
	}

	again, err := gundam.EmbedReference(card)
	if err != nil {
		t.Fatalf("EmbedReference again: %v", err)
	}

	if dot(b, again) < 0.9999 {
		t.Error("embedding is not deterministic across calls")
	}

	if err := gundam.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if _, err := pipe.EmbedReference(card); !errors.Is(err, ErrClosed) {
		t.Errorf("parent after Close: err = %v, want ErrClosed", err)
	}

	if err := pipe.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

// TestRecognizeWithoutDetector checks that an embedding-only pipeline reports
// ErrNoDetector instead of crashing when asked to detect.
func TestRecognizeWithoutDetector(t *testing.T) {
	const embedderPath = "testdata/private/models/embedder.onnx"
	if _, err := os.Stat(embedderPath); err != nil {
		t.Skipf("embedder model not available: %v", err)
	}

	pipe, err := New(Config{ORTLibPath: ortLibPath(t), EmbedderPath: embedderPath})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = pipe.Close() }()

	if _, err := pipe.Detect(splitCard(480, 670)); !errors.Is(err, ErrNoDetector) {
		t.Errorf("Detect: err = %v, want ErrNoDetector", err)
	}
}

// splitCard draws a card whose regions differ, so different artwork windows
// see different content: vertical color bands plus a horizontal gradient.
func splitCard(w, h int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			band := uint8(80 * (y * 4 / h))
			img.Set(x, y, color.RGBA{R: band, G: uint8(255 * x / w), B: 255 - band, A: 255})
		}
	}

	return img
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}

	return s
}
