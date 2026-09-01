package tcgvision

import "testing"

// rotateQuad turns a quad's corners about the origin, in image coordinates
// (y down), to build the same physical card seen at another angle.
func rotateQuad(quad [4][2]float32, quarterTurns int) [4][2]float32 {
	out := quad
	for range ((quarterTurns % 4) + 4) % 4 {
		for i, p := range out {
			out[i] = [2]float32{-p[1], p[0]}
		}
	}

	return out
}

func TestOrientQuad(t *testing.T) {
	// An upright card: corners clockwise from top-left, short edge first
	// (the shape the detector emits for a card standing up in the frame).
	upright := [4][2]float32{{100, 200}, {200, 200}, {200, 350}, {100, 350}}

	tests := []struct {
		name string
		in   [4][2]float32
		want [4][2]float32
	}{
		{
			name: "upright card is already ordered",
			in:   upright,
			want: upright,
		},
		{
			name: "corners starting elsewhere on the boundary rotate back",
			in:   [4][2]float32{{200, 350}, {100, 350}, {100, 200}, {200, 200}},
			want: upright,
		},
		{
			name: "counter-clockwise corners are rewound, not mirrored",
			in:   [4][2]float32{{100, 200}, {100, 350}, {200, 350}, {200, 200}},
			want: upright,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := orientQuad(tt.in); got != tt.want {
				t.Fatalf("orientQuad() = %v, want %v", got, tt.want)
			}
		})
	}
}

// A card lying sideways is the case a sum/difference heuristic gets wrong: the
// detector swaps the box's width and height, so the leading edge is the card's
// long axis. Whichever way it lies, the result must be rectified across the
// short edge — the artwork window is only in the right place then.
func TestOrientQuadKeepsShortEdgeLeading(t *testing.T) {
	upright := [4][2]float32{{100, 200}, {200, 200}, {200, 350}, {100, 350}}

	for _, turns := range []int{0, 1, 2, 3} {
		got := orientQuad(rotateQuad(upright, turns))

		short := edgeLen(got[0], got[1])
		long := edgeLen(got[1], got[2])

		if short >= long {
			t.Fatalf("%d quarter turns: leading edge %.1f is not the short one (%.1f)", turns, short, long)
		}

		if signedArea(got) <= 0 {
			t.Fatalf("%d quarter turns: corners wound counter-clockwise", turns)
		}
	}
}

func TestFlipQuad(t *testing.T) {
	quad := [4][2]float32{{100, 200}, {200, 200}, {200, 350}, {100, 350}}

	flipped := flipQuad(quad)
	if flipped[0] != quad[2] || flipped[2] != quad[0] {
		t.Fatalf("flipQuad = %v, want the same corners read from the opposite one", flipped)
	}

	if got := flipQuad(flipped); got != quad {
		t.Fatalf("flipping twice = %v, want the original %v", got, quad)
	}

	// A 180° turn is still the same rectangle, wound the same way.
	if signedArea(flipped) != signedArea(quad) {
		t.Fatalf("flipQuad changed the winding")
	}
}

func TestTopSim(t *testing.T) {
	if got := topSim(nil); got != 0 {
		t.Fatalf("topSim(nil) = %v, want 0", got)
	}

	if got := topSim([]Match{{Sim: 0.9}, {Sim: 0.8}}); got != 0.9 {
		t.Fatalf("topSim = %v, want 0.9", got)
	}
}
