package tcgvision

import "testing"

func TestSortReadingOrder(t *testing.T) {
	// Boxes are built from a center point with a fixed 0.24 x 0.30 size
	// (coordinates are unit-free — the sort only compares them).
	box := func(cx, cy float32) Detection {
		w, h := float32(0.12), float32(0.15)

		return Detection{Box: Box{Poly: [4][2]float32{
			{cx - w, cy - h}, {cx + w, cy - h}, {cx + w, cy + h}, {cx - w, cy + h},
		}}}
	}

	cases := []struct {
		name string
		in   []Detection
		want [][2]float32 // expected center sequence
	}{
		{
			name: "shuffled 3x3 grid with slight y jitter",
			in: []Detection{
				box(0.50, 0.51), box(0.80, 0.19), box(0.20, 0.80),
				box(0.80, 0.81), box(0.20, 0.21), box(0.50, 0.79),
				box(0.80, 0.50), box(0.20, 0.49), box(0.50, 0.20),
			},
			want: [][2]float32{
				{0.20, 0.21}, {0.50, 0.20}, {0.80, 0.19},
				{0.20, 0.49}, {0.50, 0.51}, {0.80, 0.50},
				{0.20, 0.80}, {0.50, 0.79}, {0.80, 0.81},
			},
		},
		{
			name: "single box unchanged",
			in:   []Detection{box(0.5, 0.5)},
			want: [][2]float32{{0.5, 0.5}},
		},
		{
			name: "tilted 2x2 grid",
			in:   []Detection{box(0.7, 0.33), box(0.3, 0.72), box(0.7, 0.78), box(0.3, 0.27)},
			want: [][2]float32{{0.3, 0.27}, {0.7, 0.33}, {0.3, 0.72}, {0.7, 0.78}},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			sortReadingOrder(c.in)

			for i, w := range c.want {
				var cx, cy float32
				for _, pt := range c.in[i].Poly {
					cx += pt[0] / 4
					cy += pt[1] / 4
				}
				if diff := (cx-w[0])*(cx-w[0]) + (cy-w[1])*(cy-w[1]); diff > 1e-6 {
					t.Fatalf("pos %d: center = (%.2f, %.2f), want (%.2f, %.2f)", i, cx, cy, w[0], w[1])
				}
			}
		})
	}
}
