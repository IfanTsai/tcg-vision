package tcgvision

import (
	"math"
	"path/filepath"
	"testing"
)

func TestF16Roundtrip(t *testing.T) {
	cases := []float32{0, 1, -1, 0.5, -0.03125, 0.999, 65504, 1e-6, -1e-6, 0.123456}
	for _, f := range cases {
		got := f16ToFloat32(float32ToF16(f))
		diff := math.Abs(float64(got - f))
		rel := diff / math.Max(math.Abs(float64(f)), 1e-9)
		if f != 0 && rel > 1e-3 && diff > 1e-7 {
			t.Errorf("f16 roundtrip %v -> %v (rel %v)", f, got, rel)
		}
	}
}

func TestOrientQuadLegacyOrdering(t *testing.T) {
	// The counter-clockwise quad the old sum/difference ordering was tested
	// with: orientQuad must keep producing the identical result for it.
	in := [4][2]float32{{100, 10}, {10, 12}, {12, 200}, {102, 198}}
	got := orientQuad(in)
	want := [4][2]float32{{10, 12}, {100, 10}, {102, 198}, {12, 200}}
	if got != want {
		t.Errorf("orientQuad = %v, want %v", got, want)
	}
}

func TestHomographyIdentityAndScale(t *testing.T) {
	src := [4][2]float32{{0, 0}, {100, 0}, {100, 200}, {0, 200}}
	h, err := homography(src, src)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{1, 0, 0, 0, 1, 0, 0, 0, 1} {
		if math.Abs(h[i]-want) > 1e-9 {
			t.Fatalf("identity homography[%d] = %v", i, h[i])
		}
	}

	dst := [4][2]float32{{0, 0}, {50, 0}, {50, 100}, {0, 100}}
	h, err = homography(src, dst)
	if err != nil {
		t.Fatal(err)
	}
	// maps (100, 200) -> (50, 100)
	x := (h[0]*100 + h[1]*200 + h[2]) / (h[6]*100 + h[7]*200 + h[8])
	y := (h[3]*100 + h[4]*200 + h[5]) / (h[6]*100 + h[7]*200 + h[8])
	if math.Abs(x-50) > 1e-6 || math.Abs(y-100) > 1e-6 {
		t.Errorf("scale homography maps to (%v, %v)", x, y)
	}
}

func TestPolyArea(t *testing.T) {
	quad := [4][2]float32{{0, 0}, {10, 0}, {10, 5}, {0, 5}}
	if got := polyArea(quad); got != 50 {
		t.Errorf("polyArea = %v, want 50", got)
	}
}

func TestLetterboxGeometry(t *testing.T) {
	src := newRGBImage(960, 1280)
	img, scale, padX, padY := letterbox(src, 640)
	if img.w != 640 || img.h != 640 {
		t.Fatalf("letterbox size %dx%d", img.w, img.h)
	}
	if math.Abs(float64(scale)-0.5) > 1e-6 || padX != 80 || padY != 0 {
		t.Errorf("scale %v padX %d padY %d, want 0.5 80 0", scale, padX, padY)
	}
	if img.r[0] != 114 || img.r[len(img.r)-1] != 114 {
		t.Error("padding not 114")
	}
}

func TestResizeBilinearConstant(t *testing.T) {
	src := newRGBImage(7, 5)
	for i := range src.r {
		src.r[i], src.g[i], src.b[i] = 42, 43, 44
	}
	dst := resizeBilinear(src, 3, 9)
	for i := range dst.r {
		if dst.r[i] != 42 || dst.g[i] != 43 || dst.b[i] != 44 {
			t.Fatalf("constant image changed at %d: %v %v %v", i, dst.r[i], dst.g[i], dst.b[i])
		}
	}
}

func TestIndexSaveLoadSearch(t *testing.T) {
	idx := NewIndex(4)
	vecs := map[string][]float32{
		"a": {1, 0, 0, 0},
		"b": {0, 1, 0, 0},
		"c": {0.707, 0.707, 0, 0},
	}
	for k, v := range vecs {
		if err := idx.Add(k, v); err != nil {
			t.Fatal(err)
		}
	}
	// replace must not duplicate
	if err := idx.Add("a", []float32{0, 0, 1, 0}); err != nil {
		t.Fatal(err)
	}
	if idx.Len() != 3 {
		t.Fatalf("Len = %d after replace", idx.Len())
	}

	path := filepath.Join(t.TempDir(), "idx.bin")
	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadIndex(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 3 || loaded.Dim() != 4 {
		t.Fatalf("loaded %d/%d", loaded.Len(), loaded.Dim())
	}

	got := loaded.Search([]float32{1, 0, 0, 0}, 2)
	if len(got) != 2 || got[0].Key != "c" {
		t.Fatalf("search top = %+v", got)
	}
	if math.Abs(float64(got[0].Sim)-0.707) > 1e-3 {
		t.Errorf("sim = %v", got[0].Sim)
	}

	if loaded.Search([]float32{1, 0, 0}, 2) != nil {
		t.Error("dimension-mismatched search must return nil")
	}
}

func TestDecodeOBBAndNMS(t *testing.T) {
	// two overlapping boxes (one lower-conf duplicate) + one below threshold
	n := 3
	raw := make([]float32, 6*n)
	set := func(i int, cx, cy, w, h, conf, ang float32) {
		raw[i], raw[n+i], raw[2*n+i], raw[3*n+i], raw[4*n+i], raw[5*n+i] = cx, cy, w, h, conf, ang
	}
	set(0, 320, 320, 100, 150, 0.9, 0)
	set(1, 322, 318, 100, 150, 0.6, 0.05)
	set(2, 100, 100, 50, 50, 0.1, 0)

	prof := YuGiOh()
	boxes := decodeOBB(raw, n, 1, 0, 0, prof)
	if len(boxes) != 1 {
		t.Fatalf("got %d boxes, want 1 (NMS + conf threshold)", len(boxes))
	}
	if boxes[0].Conf != 0.9 {
		t.Errorf("kept conf %v", boxes[0].Conf)
	}
	// axis-aligned 100x150 box centered at (320,320)
	want := [4][2]float32{{270, 245}, {370, 245}, {370, 395}, {270, 395}}
	got := orientQuad(boxes[0].Poly)
	for i := range want {
		if math.Abs(float64(got[i][0]-want[i][0])) > 1e-3 || math.Abs(float64(got[i][1]-want[i][1])) > 1e-3 {
			t.Fatalf("poly[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestFlatAspectOK(t *testing.T) {
	prof := YuGiOh() // card aspect 688/472 ≈ 1.458, tol ±5%

	cases := []struct {
		name string
		w, h int
		tol  float32
		want bool
	}{
		{"canonical card", 472, 688, 0.05, true},
		{"official render", 480, 711, 0.05, true},
		{"upper edge inside", 1000, 1525, 0.05, true},
		{"4:3 photo", 3000, 4000, 0.05, false},
		{"16:9 photo", 1080, 1920, 0.05, false},
		{"landscape card", 688, 472, 0.05, false},
		{"square", 500, 500, 0.05, false},
		{"disabled", 472, 688, 0, false},
		{"zero size", 0, 688, 0.05, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := prof
			p.FlatAspectTol = c.tol
			if got := flatAspectOK(c.w, c.h, p); got != c.want {
				t.Errorf("flatAspectOK(%d, %d, tol=%v) = %v, want %v", c.w, c.h, c.tol, got, c.want)
			}
		})
	}
}

func TestCenterCardCrop(t *testing.T) {
	prof := YuGiOh()
	cases := []struct {
		name string
		w, h int
	}{
		{"4:3 landscape", 1600, 1200},
		{"4:3 portrait", 1200, 1600},
		{"16:9 portrait", 900, 1600},
		{"square", 1000, 1000},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			x0, y0, x1, y1 := centerCardCrop(c.w, c.h, prof)

			if x0 < 0 || y0 < 0 || x1 > c.w || y1 > c.h {
				t.Fatalf("crop (%d,%d)-(%d,%d) leaves the %dx%d image", x0, y0, x1, y1, c.w, c.h)
			}
			if x0 != c.w-x1 && x0 != c.w-x1-1 || y0 != c.h-y1 && y0 != c.h-y1-1 {
				t.Errorf("crop (%d,%d)-(%d,%d) is not centered in %dx%d", x0, y0, x1, y1, c.w, c.h)
			}
			if x1-x0 != c.w && y1-y0 != c.h {
				t.Errorf("crop (%d,%d)-(%d,%d) touches neither side pair of %dx%d", x0, y0, x1, y1, c.w, c.h)
			}
			if !flatAspectOK(x1-x0, y1-y0, prof) {
				t.Errorf("crop %dx%d is not card-shaped", x1-x0, y1-y0)
			}
		})
	}
}
