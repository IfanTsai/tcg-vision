// Package tcgvision recognizes trading cards in photos: an oriented-box
// detector finds cards, each card is perspective-rectified, its artwork
// window is embedded with a vision transformer, and the embedding is matched
// against an index of reference card images by cosine similarity.
//
// The pipeline is game-agnostic; game specifics (detector model, card aspect,
// artwork window) live in a Profile. Models are ONNX and run on CPU through
// onnxruntime; callers provide the paths to the onnxruntime shared library
// and the two model files.
package tcgvision

import (
	"fmt"
	"image"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// EmbedDim is the embedding dimension produced by the embedder model
// (CLS + mean-patch concat of a ViT-S backbone).
const EmbedDim = 768

// imagenet normalization constants used by the embedder.
var (
	embedMean = [3]float32{0.485, 0.456, 0.406}
	embedStd  = [3]float32{0.229, 0.224, 0.225}
)

// embedInputSize is the square input size of the embedder model.
const embedInputSize = 224

// Config configures a Pipeline.
type Config struct {
	// ORTLibPath is the path to libonnxruntime.so; required once per process.
	ORTLibPath string
	// DetectorPath is the ONNX oriented-box card detector.
	DetectorPath string
	// EmbedderPath is the ONNX artwork embedder.
	EmbedderPath string
	// Threads caps intra-op parallelism per inference (default 2).
	Threads int
	// Profile holds the game-specific parameters (default YuGiOh()).
	Profile Profile
}

// Pipeline runs detection, rectification and embedding. Inference calls are
// serialized internally so a Pipeline has a bounded memory footprint.
type Pipeline struct {
	mu       sync.Mutex
	detector *ort.DynamicAdvancedSession
	embedder *ort.DynamicAdvancedSession
	profile  Profile
}

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

// Detection is one recognized card: where it is and what it looks like.
type Detection struct {
	Box
	// Flat marks the whole-image fallback (see Profile.FlatAspectTol): no card
	// was detected but the image itself has a card-like aspect ratio, so it
	// was embedded as one upright card. Poly covers the full image and Conf is
	// 0 (there is no detector confidence to report).
	Flat    bool
	Matches []Match
}

// New loads both models and returns a ready Pipeline.
func New(cfg Config) (*Pipeline, error) {
	if cfg.Profile.DetectSize == 0 {
		cfg.Profile = YuGiOh()
	}
	if cfg.Threads <= 0 {
		cfg.Threads = 2
	}

	if err := initORT(cfg.ORTLibPath); err != nil {
		return nil, fmt.Errorf("initialize onnxruntime: %w", err)
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	defer func() { _ = opts.Destroy() }()
	if err := opts.SetIntraOpNumThreads(cfg.Threads); err != nil {
		return nil, fmt.Errorf("set intra-op threads: %w", err)
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("set inter-op threads: %w", err)
	}

	var detector *ort.DynamicAdvancedSession
	if cfg.DetectorPath != "" {
		detector, err = ort.NewDynamicAdvancedSession(cfg.DetectorPath, []string{"images"}, []string{"output0"}, opts)
		if err != nil {
			return nil, fmt.Errorf("load detector %s: %w", cfg.DetectorPath, err)
		}
	}

	embedder, err := ort.NewDynamicAdvancedSession(cfg.EmbedderPath, []string{"pixel_values"}, []string{"emb"}, opts)
	if err != nil {
		_ = detector.Destroy()

		return nil, fmt.Errorf("load embedder %s: %w", cfg.EmbedderPath, err)
	}

	return &Pipeline{detector: detector, embedder: embedder, profile: cfg.Profile}, nil
}

// Profile returns the pipeline's game profile.
func (p *Pipeline) Profile() Profile { return p.profile }

// Detect finds cards in a photo and returns their quads in photo coordinates,
// filtered by the profile's confidence and minimum-area rules.
func (p *Pipeline) Detect(img image.Image) ([]Box, error) {
	return p.detectRGB(fromImage(img))
}

// Recognize runs the full pipeline on a photo: detect cards, embed each one,
// and return the topK index matches per card. When nothing is detected and
// the image itself has a card-like aspect ratio, the whole image is embedded
// as one upright card instead (Detection.Flat).
//
// Pass the photo at full resolution: photos larger than Profile.MaxPhotoSide
// are downscaled internally, and coordinates in the results are always in the
// input photo's pixel space. Detections are returned in reading order (rows
// top to bottom, left to right within a row).
func (p *Pipeline) Recognize(img image.Image, idx *Index, topK int) ([]Detection, error) {
	scaled := img
	if p.profile.MaxPhotoSide > 0 {
		scaled = Downscale(img, p.profile.MaxPhotoSide)
	}
	rgb := fromImage(scaled)

	dets, err := p.recognizeRGB(rgb, idx, topK)
	if err != nil || len(dets) == 0 {
		return dets, err
	}

	// The pipeline ran on the downscaled photo: map coordinates back to the
	// input photo's pixel space (per-axis, exact under rounding).
	sx := float32(img.Bounds().Dx()) / float32(rgb.w)
	sy := float32(img.Bounds().Dy()) / float32(rgb.h)
	if sx != 1 || sy != 1 {
		for d := range dets {
			for i := range dets[d].Poly {
				dets[d].Poly[i][0] *= sx
				dets[d].Poly[i][1] *= sy
			}
		}
	}

	sortReadingOrder(dets)

	return dets, nil
}

// EmbedCards rectifies each detected box and embeds its artwork window in one
// batch. Order matches boxes.
func (p *Pipeline) EmbedCards(img image.Image, boxes []Box) ([][]float32, error) {
	if len(boxes) == 0 {
		return nil, nil
	}

	rgb := fromImage(img)
	prof := p.profile
	batch := make([]float32, 0, len(boxes)*3*embedInputSize*embedInputSize)
	for _, b := range boxes {
		card, err := warpCard(rgb, orientQuad(b.Poly), prof.CardW, prof.CardH)
		if err != nil {
			return nil, fmt.Errorf("rectify card: %w", err)
		}
		batch = embedInput(batch, artWindow(card, prof))
	}

	return p.runEmbedder(batch, len(boxes))
}

// EmbedReference embeds a flat reference card image (an official render or a
// clean printing photo) for indexing: same artwork window, no rectification.
func (p *Pipeline) EmbedReference(img image.Image) ([]float32, error) {
	rgb := fromImage(img)
	batch := embedInput(nil, artWindow(rgb, p.profile))

	vecs, err := p.runEmbedder(batch, 1)
	if err != nil {
		return nil, err
	}

	return vecs[0], nil
}

// Close releases the model sessions.
func (p *Pipeline) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.detector != nil {
		_ = p.detector.Destroy()
		p.detector = nil
	}
	if p.embedder != nil {
		_ = p.embedder.Destroy()
		p.embedder = nil
	}

	return nil
}

func (p *Pipeline) detectRGB(rgb *rgbImage) ([]Box, error) {
	prof := p.profile
	lb, scale, padX, padY := letterbox(rgb, prof.DetectSize)

	input, err := ort.NewTensor(ort.NewShape(1, 3, int64(prof.DetectSize), int64(prof.DetectSize)), detectInput(lb))
	if err != nil {
		return nil, fmt.Errorf("create detector input: %w", err)
	}
	defer func() { _ = input.Destroy() }()

	p.mu.Lock()
	outputs := []ort.Value{nil}
	err = p.detector.Run([]ort.Value{input}, outputs)
	p.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("run detector: %w", err)
	}
	defer func() { _ = outputs[0].Destroy() }()

	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected detector output type %T", outputs[0])
	}
	shape := out.GetShape()
	if len(shape) != 3 || shape[1] != 6 {
		return nil, fmt.Errorf("unexpected detector output shape %v", shape)
	}

	boxes := decodeOBB(out.GetData(), int(shape[2]), scale, padX, padY, prof)

	photoArea := float32(rgb.w) * float32(rgb.h)
	kept := boxes[:0]
	for _, b := range boxes {
		if b.Conf >= prof.MinConf && polyArea(b.Poly) >= prof.MinAreaFrac*photoArea {
			kept = append(kept, b)
		}
	}

	return kept, nil
}

// runEmbedder embeds a batch of pre-packed inputs, returning one normalized
// EmbedDim vector per item (the model output is already L2-normalized).
func (p *Pipeline) runEmbedder(batch []float32, count int) ([][]float32, error) {
	input, err := ort.NewTensor(ort.NewShape(int64(count), 3, embedInputSize, embedInputSize), batch)
	if err != nil {
		return nil, fmt.Errorf("create embedder input: %w", err)
	}
	defer func() { _ = input.Destroy() }()

	p.mu.Lock()
	outputs := []ort.Value{nil}
	err = p.embedder.Run([]ort.Value{input}, outputs)
	p.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("run embedder: %w", err)
	}
	defer func() { _ = outputs[0].Destroy() }()

	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected embedder output type %T", outputs[0])
	}
	data := out.GetData()
	if len(data) != count*EmbedDim {
		return nil, fmt.Errorf("unexpected embedder output size %d", len(data))
	}

	vecs := make([][]float32, count)
	for i := range count {
		vec := make([]float32, EmbedDim)
		copy(vec, data[i*EmbedDim:(i+1)*EmbedDim])
		vecs[i] = vec
	}

	return vecs, nil
}

// recognizeRGB is Recognize on the (already bounded) working image.
func (p *Pipeline) recognizeRGB(rgb *rgbImage, idx *Index, topK int) ([]Detection, error) {
	boxes, err := p.detectRGB(rgb)
	if err != nil {
		return nil, err
	}
	if len(boxes) == 0 {
		return p.recognizeFlat(rgb, idx, topK)
	}

	prof := p.profile
	quads := make([][4][2]float32, len(boxes))
	batch := make([]float32, 0, len(boxes)*3*embedInputSize*embedInputSize)

	for i, b := range boxes {
		quads[i] = orientQuad(b.Poly)

		card, err := warpCard(rgb, quads[i], prof.CardW, prof.CardH)
		if err != nil {
			return nil, fmt.Errorf("rectify card: %w", err)
		}

		batch = embedInput(batch, artWindow(card, prof))
	}

	vecs, err := p.runEmbedder(batch, len(boxes))
	if err != nil {
		return nil, err
	}

	dets := make([]Detection, len(boxes))
	for i, b := range boxes {
		dets[i] = Detection{Box: b, Matches: idx.Search(vecs[i], topK)}
	}

	if err := p.retryFlipped(rgb, quads, dets, idx, topK); err != nil {
		return nil, err
	}

	return dets, nil
}

// retryFlipped re-embeds the 180° twin of every card whose match came back
// weak and keeps the better of the two. Rectification pins a card's upright
// direction only to within 180° (see orientQuad and Profile.FlipRetryBelowSim);
// a card rectified upside down scores middling against some unrelated card
// instead of failing outright, so a weak match is the signal to try the twin.
// Cards that already matched well — the common case — cost nothing extra.
func (p *Pipeline) retryFlipped(
	rgb *rgbImage, quads [][4][2]float32, dets []Detection, idx *Index, topK int,
) error {
	prof := p.profile
	if prof.FlipRetryBelowSim <= 0 {
		return nil
	}

	var weak []int

	for i := range dets {
		if topSim(dets[i].Matches) < prof.FlipRetryBelowSim {
			weak = append(weak, i)
		}
	}

	if len(weak) == 0 {
		return nil
	}

	batch := make([]float32, 0, len(weak)*3*embedInputSize*embedInputSize)

	for _, i := range weak {
		card, err := warpCard(rgb, flipQuad(quads[i]), prof.CardW, prof.CardH)
		if err != nil {
			return fmt.Errorf("rectify flipped card: %w", err)
		}

		batch = embedInput(batch, artWindow(card, prof))
	}

	vecs, err := p.runEmbedder(batch, len(weak))
	if err != nil {
		return err
	}

	for j, i := range weak {
		if matches := idx.Search(vecs[j], topK); topSim(matches) > topSim(dets[i].Matches) {
			dets[i].Matches = matches
		}
	}

	return nil
}

// topSim is the best similarity in a match list (0 when empty).
func topSim(matches []Match) float32 {
	if len(matches) == 0 {
		return 0
	}

	return matches[0].Sim
}

// recognizeFlat is the no-detection fallback of Recognize: flat scans and
// official renders score near zero on the photo-trained detector, so a
// card-aspect image is embedded whole as one upright card. Returns nil when
// the fallback is disabled or the aspect ratio does not fit.
func (p *Pipeline) recognizeFlat(rgb *rgbImage, idx *Index, topK int) ([]Detection, error) {
	prof := p.profile
	if !flatAspectOK(rgb.w, rgb.h, prof) {
		return nil, nil
	}

	vecs, err := p.runEmbedder(embedInput(nil, artWindow(rgb, prof)), 1)
	if err != nil {
		return nil, err
	}

	w, h := float32(rgb.w), float32(rgb.h)
	det := Detection{
		Box:     Box{Poly: [4][2]float32{{0, 0}, {w, 0}, {w, h}, {0, h}}},
		Flat:    true,
		Matches: idx.Search(vecs[0], topK),
	}

	return []Detection{det}, nil
}

// initORT initializes the process-wide onnxruntime environment once.
func initORT(libPath string) error {
	ortInitOnce.Do(func() {
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		ortInitErr = ort.InitializeEnvironment()
	})

	return ortInitErr
}

// embedInput packs an artwork crop (resized to the embedder input size) into
// a normalized NCHW tensor slice, appended to dst.
func embedInput(dst []float32, art *rgbImage) []float32 {
	resized := resizeBilinear(art, embedInputSize, embedInputSize)
	n := embedInputSize * embedInputSize
	for i := range n {
		dst = append(dst, (resized.r[i]/255-embedMean[0])/embedStd[0])
	}

	for i := range n {
		dst = append(dst, (resized.g[i]/255-embedMean[1])/embedStd[1])
	}

	for i := range n {
		dst = append(dst, (resized.b[i]/255-embedMean[2])/embedStd[2])
	}

	return dst
}

// artWindow crops the profile's artwork window from an upright card image.
func artWindow(card *rgbImage, prof Profile) *rgbImage {
	x0 := int(float32(card.w) * prof.ArtX)
	y0 := int(float32(card.h) * prof.ArtY)
	x1 := int(float32(card.w) * (prof.ArtX + prof.ArtW))
	y1 := int(float32(card.h) * (prof.ArtY + prof.ArtH))

	return card.crop(x0, y0, x1, y1)
}

// flatAspectOK reports whether an image of the given size is close enough to
// the profile's card aspect ratio to be treated as one flat upright card.
func flatAspectOK(w, h int, prof Profile) bool {
	if prof.FlatAspectTol <= 0 || w <= 0 || h <= 0 {
		return false
	}

	aspect := float32(h) / float32(w)
	want := float32(prof.CardH) / float32(prof.CardW)

	return aspect >= want*(1-prof.FlatAspectTol) && aspect <= want*(1+prof.FlatAspectTol)
}
