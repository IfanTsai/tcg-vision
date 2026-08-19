package tcgvision

import "sort"

// sortReadingOrder sorts detections into human reading order: rows top to
// bottom, left to right within a row. Rows are built by clustering box
// centers on y — scanning in ascending center-y, a box starts a new row when
// its center is more than half the average box height below the current
// row's first box, which tolerates the slight tilt of handheld photos.
// Explicit clustering (rather than a pairwise comparator) avoids the
// ambiguity of a non-transitive sort.
func sortReadingOrder(dets []Detection) {
	if len(dets) < 2 {
		return
	}

	type placed struct {
		det    Detection
		cx, cy float32
	}

	items := make([]placed, len(dets))
	var avgH float32
	for i, d := range dets {
		var cx, cy float32
		minY, maxY := d.Poly[0][1], d.Poly[0][1]
		for _, pt := range d.Poly {
			cx += pt[0] / 4
			cy += pt[1] / 4
			minY = min(minY, pt[1])
			maxY = max(maxY, pt[1])
		}

		items[i] = placed{det: d, cx: cx, cy: cy}
		avgH += maxY - minY
	}
	avgH /= float32(len(dets))

	sort.SliceStable(items, func(i, j int) bool { return items[i].cy < items[j].cy })

	rows := [][]placed{}
	for _, it := range items {
		if len(rows) > 0 && it.cy-rows[len(rows)-1][0].cy <= avgH/2 {
			rows[len(rows)-1] = append(rows[len(rows)-1], it)

			continue
		}

		rows = append(rows, []placed{it})
	}

	i := 0
	for _, row := range rows {
		sort.SliceStable(row, func(a, b int) bool { return row[a].cx < row[b].cx })
		for _, it := range row {
			dets[i] = it.det
			i++
		}
	}
}
