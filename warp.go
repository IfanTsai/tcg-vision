package tcgvision

import (
	"fmt"
	"math"
)

// orderQuad orders four polygon points as [top-left, top-right, bottom-right,
// bottom-left] using the sum/difference heuristic (valid for card-like quads
// that are not rotated beyond ±45°).
func orderQuad(poly [4][2]float32) [4][2]float32 {
	var tl, tr, br, bl [2]float32
	minSum, maxSum := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	minDiff, maxDiff := float32(math.MaxFloat32), float32(-math.MaxFloat32)

	for _, p := range poly {
		sum := p[0] + p[1]
		diff := p[1] - p[0]
		if sum < minSum {
			minSum, tl = sum, p
		}
		if sum > maxSum {
			maxSum, br = sum, p
		}
		if diff < minDiff {
			minDiff, tr = diff, p
		}
		if diff > maxDiff {
			maxDiff, bl = diff, p
		}
	}

	return [4][2]float32{tl, tr, br, bl}
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
func warpCard(img *rgbImage, quad [4][2]float32, cardW, cardH int) (*rgbImage, error) {
	src := orderQuad(quad)
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
