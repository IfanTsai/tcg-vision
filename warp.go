package tcgvision

import (
	"fmt"
	"math"
)

// orientQuad orders four detected corners as [top-left, top-right,
// bottom-right, bottom-left] of the upright card.
//
// The detector emits corners in boundary order and swaps the box's width and
// height for a card lying sideways in the frame, so the two adjacent edge
// lengths are what separates the card's long axis from its short one: the
// short edge is the card's width. Rotating the sequence until a short edge
// comes first is exactly what a plain sum/difference heuristic cannot do —
// that heuristic only holds within ±45°, and a sideways card (the natural
// shape of a photo of an open binder page) then gets rectified across its
// short axis, putting the artwork window on the card's flank instead of the
// art. The embedding that comes out matches some *other* card at middling
// similarity, which is worse than returning no match at all.
//
// Geometry cannot settle the remaining 180°: the two candidates are mirror
// images and only the artwork says which way is up. This picks the corner
// nearest the image origin, which is right for a roughly upright card (and
// keeps those identical to the older ordering) and a coin flip otherwise —
// Recognize sorts the rest out by embedding the 180° twin of any card whose
// match came back weak (Profile.FlipRetryBelowSim).
//
// The short-edge rule assumes the profile's cards are clearly non-square
// (Yu-Gi-Oh! is 59x86mm). For a near-square card the two edge lengths differ
// by less than perspective noise and the 90° choice degrades to chance —
// a profile for such a game would need an orientation cue of its own.
func orientQuad(quad [4][2]float32) [4][2]float32 {
	// Winding decides whether the rectified card comes out as the card or as
	// its mirror image, and rotating the sequence preserves whatever winding
	// came in. The detector always emits clockwise corners; a caller building
	// boxes by hand may not.
	if signedArea(quad) < 0 {
		quad = [4][2]float32{quad[0], quad[3], quad[2], quad[1]}
	}

	start := 0
	if edgeLen(quad[0], quad[1]) > edgeLen(quad[1], quad[2]) {
		start = 1
	}

	// Of the two rotations that put a short edge first, take the one whose
	// leading corner sits nearest the origin.
	if opposite := (start + 2) % 4; corner(quad[opposite]) < corner(quad[start]) {
		start = opposite
	}

	var out [4][2]float32
	for i := range 4 {
		out[i] = quad[(start+i)%4]
	}

	return out
}

// flipQuad reads the same corners starting from the opposite one: a card
// rectified from the result comes out rotated by 180°.
func flipQuad(quad [4][2]float32) [4][2]float32 {
	return [4][2]float32{quad[2], quad[3], quad[0], quad[1]}
}

// signedArea is the shoelace area of a quad: positive when its corners wind
// clockwise in image coordinates (y down).
func signedArea(quad [4][2]float32) float32 {
	var sum float32

	for i := range 4 {
		j := (i + 1) % 4
		sum += quad[i][0]*quad[j][1] - quad[j][0]*quad[i][1]
	}

	return sum / 2
}

// corner ranks a point by its distance from the image origin along both axes.
func corner(p [2]float32) float32 { return p[0] + p[1] }

// edgeLen is the distance between two corners.
func edgeLen(a, b [2]float32) float64 {
	return math.Hypot(float64(a[0]-b[0]), float64(a[1]-b[1]))
}

// homography solves the 3x3 projective transform H mapping src[i] -> dst[i]
// for four point pairs (h22 fixed to 1), via Gaussian elimination on the
// standard 8x8 system.
func homography(src, dst [4][2]float32) ([9]float64, error) {
	var a [8][9]float64 // augmented matrix
	for i := range 4 {
		sx, sy := float64(src[i][0]), float64(src[i][1])
		dx, dy := float64(dst[i][0]), float64(dst[i][1])
		a[2*i] = [9]float64{sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy, dx}
		a[2*i+1] = [9]float64{0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy, dy}
	}

	for col := range 8 {
		pivot := col
		for row := col + 1; row < 8; row++ {
			if math.Abs(a[row][col]) > math.Abs(a[pivot][col]) {
				pivot = row
			}
		}
		if math.Abs(a[pivot][col]) < 1e-12 {
			return [9]float64{}, fmt.Errorf("degenerate quad")
		}
		a[col], a[pivot] = a[pivot], a[col]

		for row := range 8 {
			if row == col {
				continue
			}
			f := a[row][col] / a[col][col]
			for k := col; k < 9; k++ {
				a[row][k] -= f * a[col][k]
			}
		}
	}

	var h [9]float64
	for i := range 8 {
		h[i] = a[i][8] / a[i][i]
	}
	h[8] = 1

	return h, nil
}

// warpCard rectifies the quad in img to an upright (cardW x cardH) card via
// the inverse homography and bilinear sampling; out-of-bounds samples clamp
// to the image edge (detection boxes hug the card, so the border effect on
// the inner artwork window is nil).
//
// The quad must already be ordered [top-left, top-right, bottom-right,
// bottom-left] for the card's upright direction — pass it through orientQuad
// (detector corners come in boundary order, which is not the same thing).
func warpCard(img *rgbImage, quad [4][2]float32, cardW, cardH int) (*rgbImage, error) {
	src := quad
	dst := [4][2]float32{{0, 0}, {float32(cardW), 0}, {float32(cardW), float32(cardH)}, {0, float32(cardH)}}

	// Solve dst -> src directly so each output pixel maps back into the photo.
	h, err := homography(dst, src)
	if err != nil {
		return nil, fmt.Errorf("solve homography: %w", err)
	}

	out := newRGBImage(cardW, cardH)
	i := 0
	for y := range cardH {
		fy := float64(y)
		for x := range cardW {
			fx := float64(x)
			den := h[6]*fx + h[7]*fy + h[8]
			sx := float32((h[0]*fx + h[1]*fy + h[2]) / den)
			sy := float32((h[3]*fx + h[4]*fy + h[5]) / den)
			out.r[i] = bilinearAt(img.r, img.w, img.h, sx, sy)
			out.g[i] = bilinearAt(img.g, img.w, img.h, sx, sy)
			out.b[i] = bilinearAt(img.b, img.w, img.h, sx, sy)
			i++
		}
	}

	return out, nil
}
