package tcgvision

import (
	"encoding/json"
	"image"
	_ "image/jpeg"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// Golden alignment against the Python reference implementation. Runs only
// when the private assets exist (testdata/private is gitignored: real photos,
// models and the reference index are not distributed with the repo).
//
// TCGVISION_ORT can override the libonnxruntime.so path.

type goldenDet struct {
	Poly    [4][2]float32 `json:"poly"`
	Conf    float32       `json:"conf"`
	EmbFull []float32     `json:"emb_full"`
	Top5    []struct {
		Key string  `json:"key"`
		Sim float32 `json:"sim"`
	} `json:"top5"`
}

func TestGoldenAgainstPythonReference(t *testing.T) {
	golden := loadPrivateGolden(t)

	pipe, err := New(Config{
		ORTLibPath:   ortLibPath(t),
		DetectorPath: "testdata/private/models/detector.onnx",
		EmbedderPath: "testdata/private/models/embedder.onnx",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = pipe.Close() }()

	idx, err := LoadIndex("testdata/private/index_art.bin")
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}
	if idx.Len() < 40000 {
		t.Fatalf("reference index unexpectedly small: %d", idx.Len())
	}

	names := make([]string, 0, len(golden))
	for name := range golden {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		want := golden[name]
		t.Run(name[:8], func(t *testing.T) {
			f, err := os.Open(filepath.Join("testdata/private/photos", name))
			if err != nil {
				t.Skipf("photo missing: %v", err)
			}
			img, _, err := image.Decode(f)
			_ = f.Close()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}

			dets, err := pipe.Recognize(img, idx, 5)
			if err != nil {
				t.Fatalf("Recognize: %v", err)
			}
			if len(dets) != len(want) {
				t.Fatalf("got %d detections, golden has %d", len(dets), len(want))
			}

			vecs, err := pipe.EmbedCards(img, boxesOf(dets))
			if err != nil {
				t.Fatalf("EmbedCards: %v", err)
			}

			for i, w := range want {
				if iou := quadIoU(dets[i].Poly, w.Poly); iou < 0.97 {
					t.Errorf("det %d IoU vs golden = %.4f", i, iou)
				}
				if d := math.Abs(float64(dets[i].Conf - w.Conf)); d > 0.03 {
					t.Errorf("det %d conf %.3f vs golden %.3f", i, dets[i].Conf, w.Conf)
				}
				if cos := cosine(vecs[i], w.EmbFull); cos < 0.99 {
					t.Errorf("det %d embedding cosine vs golden = %.5f", i, cos)
				}

				// Decoder/interpolation differences may swap near-tied
				// neighbors, so require mutual top-5 membership of the two
				// top-1 keys plus a tight similarity match, not exact order.
				if len(dets[i].Matches) == 0 {
					t.Fatalf("det %d has no matches", i)
				}
				goldenKeys := make(map[string]bool, len(w.Top5))
				for _, m := range w.Top5 {
					goldenKeys[m.Key] = true
				}
				gotKeys := make(map[string]bool, len(dets[i].Matches))
				for _, m := range dets[i].Matches {
					gotKeys[m.Key] = true
				}
				if !goldenKeys[dets[i].Matches[0].Key] || !gotKeys[w.Top5[0].Key] {
					t.Errorf("det %d top1 %s (golden %s) not mutually in top5", i, dets[i].Matches[0].Key, w.Top5[0].Key)
				}
				if d := math.Abs(float64(dets[i].Matches[0].Sim - w.Top5[0].Sim)); d > 0.02 {
					t.Errorf("det %d top1 sim %.4f vs golden %.4f", i, dets[i].Matches[0].Sim, w.Top5[0].Sim)
				}
			}
		})
	}
}

// TestRecognizeFlatFallback covers the whole-image fallback on a flat
// official render (the photo-trained detector scores it near zero): Recognize
// must return one Flat detection whose top match is the exact printing.
func TestRecognizeFlatFallback(t *testing.T) {
	loadPrivateGolden(t) // skip when private assets are absent

	pipe, err := New(Config{
		ORTLibPath:   ortLibPath(t),
		DetectorPath: "testdata/private/models/detector.onnx",
		EmbedderPath: "testdata/private/models/embedder.onnx",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = pipe.Close() }()

	idx, err := LoadIndex("testdata/private/index_art.bin")
	if err != nil {
		t.Fatalf("LoadIndex: %v", err)
	}

	f, err := os.Open("testdata/private/photos/flat-superpoly-losp.jpg")
	if err != nil {
		t.Fatalf("open flat render: %v", err)
	}
	defer func() { _ = f.Close() }()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatalf("decode flat render: %v", err)
	}

	dets, err := pipe.Recognize(img, idx, 5)
	if err != nil {
		t.Fatalf("Recognize: %v", err)
	}
	if len(dets) != 1 || !dets[0].Flat {
		t.Fatalf("want exactly one flat detection, got %+v", dets)
	}
	if len(dets[0].Matches) == 0 {
		t.Fatal("flat detection has no matches")
	}

	top := dets[0].Matches[0]
	if want := "SuperPolymerization-LOSP-JP-PScR"; !strings.Contains(top.Key, want) {
		t.Errorf("top match key = %s (sim %.4f), want it to contain %s", top.Key, top.Sim, want)
	}
	if top.Sim < 0.9 {
		t.Errorf("top match sim = %.4f, want >= 0.9", top.Sim)
	}
}

func ortLibPath(t *testing.T) string {
	if p := os.Getenv("TCGVISION_ORT"); p != "" {
		return p
	}
	matches, _ := filepath.Glob("third_party/onnxruntime-*/lib/libonnxruntime.so")
	if len(matches) == 0 {
		t.Skip("onnxruntime shared library not found (run make ort or set TCGVISION_ORT)")
	}

	return matches[0]
}

func loadPrivateGolden(t *testing.T) map[string][]goldenDet {
	raw, err := os.ReadFile("testdata/private/golden_photos.json")
	if err != nil {
		t.Skipf("private golden assets not available: %v", err)
	}

	var golden map[string][]goldenDet
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatalf("parse golden json: %v", err)
	}

	return golden
}

func quadIoU(a, b [4][2]float32) float64 {
	// axis-aligned IoU is enough: golden and got quads are near-identical
	ax0, ay0, ax1, ay1 := aabb(a)
	bx0, by0, bx1, by1 := aabb(b)
	ix := math.Min(float64(ax1), float64(bx1)) - math.Max(float64(ax0), float64(bx0))
	iy := math.Min(float64(ay1), float64(by1)) - math.Max(float64(ay0), float64(by0))
	if ix <= 0 || iy <= 0 {
		return 0
	}
	inter := ix * iy
	areaA := float64(ax1-ax0) * float64(ay1-ay0)
	areaB := float64(bx1-bx0) * float64(by1-by0)

	return inter / (areaA + areaB - inter)
}

func aabb(q [4][2]float32) (x0, y0, x1, y1 float32) {
	x0, y0 = q[0][0], q[0][1]
	x1, y1 = x0, y0
	for _, p := range q {
		x0, y0 = min(x0, p[0]), min(y0, p[1])
		x1, y1 = max(x1, p[0]), max(y1, p[1])
	}

	return
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}

	return dot / math.Sqrt(na*nb)
}

func boxesOf(dets []Detection) []Box {
	boxes := make([]Box, len(dets))
	for i, d := range dets {
		boxes[i] = d.Box
	}

	return boxes
}
