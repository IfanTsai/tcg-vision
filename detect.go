package tcgvision

import (
	"math"
	"sort"
)

// Box is one detected card: a quad in original photo coordinates ordered
// [top-left, top-right, bottom-right, bottom-left], plus detector confidence.
type Box struct {
	Poly [4][2]float32
	Conf float32
}

// letterbox scales src to fit a size x size square (keeping aspect, centered,
// gray 114 padding — the YOLO convention) and reports the scale and offsets
// needed to map detections back to source coordinates.
func letterbox(src *rgbImage, size int) (img *rgbImage, scale float32, padX, padY int) {
	scale = float32(size) / float32(src.w)
	if s := float32(size) / float32(src.h); s < scale {
		scale = s
	}
	nw := int(math.Round(float64(float32(src.w) * scale)))
	nh := int(math.Round(float64(float32(src.h) * scale)))
	padX = (size - nw) / 2
	padY = (size - nh) / 2

	resized := resizeBilinear(src, nw, nh)
	img = newRGBImage(size, size)
	for i := range img.r {
		img.r[i], img.g[i], img.b[i] = 114, 114, 114
	}
	for y := range nh {
		dst := (y+padY)*size + padX
		srcRow := y * nw
		copy(img.r[dst:dst+nw], resized.r[srcRow:srcRow+nw])
		copy(img.g[dst:dst+nw], resized.g[srcRow:srcRow+nw])
		copy(img.b[dst:dst+nw], resized.b[srcRow:srcRow+nw])
	}

	return img, scale, padX, padY
}

// detectInput packs a letterboxed image into NCHW float32 [0,1].
func detectInput(img *rgbImage) []float32 {
	n := img.w * img.h
	out := make([]float32, 3*n)
	for i := range n {
		out[i] = img.r[i] / 255
		out[n+i] = img.g[i] / 255
		out[2*n+i] = img.b[i] / 255
	}

	return out
}

// decodeOBB turns the raw detector output [1,6,N] (cx,cy,w,h,conf,angle in
// letterbox coordinates) into quads in original photo coordinates, applying
// the confidence threshold and axis-aligned NMS.
func decodeOBB(raw []float32, n int, scale float32, padX, padY int, p Profile) []Box {
	type cand struct {
		poly [4][2]float32
		conf float32
		aabb [4]float32
	}

	var cands []cand
	for i := range n {
		conf := raw[4*n+i]
		if conf < p.ConfThreshold {
			continue
		}
		cx, cy := raw[i], raw[n+i]
		w, h := raw[2*n+i], raw[3*n+i]
		ang := float64(raw[5*n+i])
		c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
		dxx, dxy := w/2*c, w/2*s
		dyx, dyy := -h/2*s, h/2*c

		var poly [4][2]float32
		signs := [4][2]float32{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}
		minX, minY := float32(math.MaxFloat32), float32(math.MaxFloat32)
		maxX, maxY := float32(-math.MaxFloat32), float32(-math.MaxFloat32)
		for k, sg := range signs {
			x := (cx + sg[0]*dxx + sg[1]*dyx - float32(padX)) / scale
			y := (cy + sg[0]*dxy + sg[1]*dyy - float32(padY)) / scale
			poly[k] = [2]float32{x, y}
			minX, minY = min(minX, x), min(minY, y)
			maxX, maxY = max(maxX, x), max(maxY, y)
		}
		cands = append(cands, cand{poly: poly, conf: conf, aabb: [4]float32{minX, minY, maxX, maxY}})
	}

	sort.Slice(cands, func(i, j int) bool { return cands[i].conf > cands[j].conf })

	var kept []cand
	for _, c := range cands {
		suppressed := false
		for _, k := range kept {
			ix := min(c.aabb[2], k.aabb[2]) - max(c.aabb[0], k.aabb[0])
			iy := min(c.aabb[3], k.aabb[3]) - max(c.aabb[1], k.aabb[1])
			if ix <= 0 || iy <= 0 {
				continue
			}
			inter := ix * iy
			areaC := (c.aabb[2] - c.aabb[0]) * (c.aabb[3] - c.aabb[1])
			areaK := (k.aabb[2] - k.aabb[0]) * (k.aabb[3] - k.aabb[1])
			if inter/(areaC+areaK-inter) > p.IoUThreshold {
				suppressed = true
				break
			}
		}
		if !suppressed {
			kept = append(kept, c)
		}
	}

	boxes := make([]Box, 0, len(kept))
	for _, c := range kept {
		boxes = append(boxes, Box{Poly: c.poly, Conf: c.conf})
	}

	return boxes
}

// polyArea returns the area of a quad via the shoelace formula.
func polyArea(poly [4][2]float32) float32 {
	var sum float32
	for i := range 4 {
		j := (i + 1) % 4
		sum += poly[i][0]*poly[j][1] - poly[j][0]*poly[i][1]
	}
	if sum < 0 {
		sum = -sum
	}

	return sum / 2
}
