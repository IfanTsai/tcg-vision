package tcgvision

// Profile holds the game-specific parameters of the pipeline: detector input
// size and thresholds, the canonical card size used for perspective
// rectification, and the artwork window that gets embedded. Everything that
// ties the pipeline to one TCG lives here; the code itself is game-agnostic.
type Profile struct {
	// DetectSize is the square letterbox size fed to the detector (pixels).
	DetectSize int
	// ConfThreshold discards raw detections below this confidence before NMS.
	ConfThreshold float32
	// IoUThreshold is the axis-aligned IoU above which NMS suppresses a box.
	IoUThreshold float32
	// MinConf discards final detections below this confidence.
	MinConf float32
	// MinAreaFrac discards final detections whose polygon area is smaller
	// than this fraction of the photo area (kills tiny false positives).
	MinAreaFrac float32
	// CardW, CardH are the canonical upright card size after rectification,
	// chosen to match the physical aspect ratio of the game's cards.
	CardW, CardH int
	// ArtX, ArtY, ArtW, ArtH describe the artwork window as fractions of the
	// canonical card. Only this window is embedded: card frames and text look
	// alike across cards (and across foil treatments), the artwork does not.
	ArtX, ArtY, ArtW, ArtH float32
	// MaxPhotoSide bounds the photo fed to Recognize: larger photos are
	// downscaled internally so their longer side is at most this many pixels
	// (coordinates in the results are still reported in the original photo's
	// pixel space). Bounds memory and latency on full-resolution camera
	// photos; detection and embedding quality are unaffected well above the
	// detector's input size. 0 disables the internal downscale.
	MaxPhotoSide int
	// FlipRetryBelowSim re-embeds a card rotated by 180° when its best index
	// match scores below this similarity, and keeps whichever orientation
	// matches better. Rectification can only pin a card's upright direction to
	// within 180° (see orientQuad): a sideways card's two candidates are
	// mirror images, and a card photographed upside down looks upright to the
	// geometry. Getting it wrong puts the artwork window on the wrong part of
	// the card, which scores middling against some unrelated card rather than
	// failing outright — measured on real photos, a sideways card lands around
	// 0.75 with the wrong orientation and 0.90 with the right one. Gating the
	// retry on a weak first match keeps the common case (an upright card, one
	// embedding) at its original cost. 0 disables the retry.
	FlipRetryBelowSim float32
	// FlatAspectTol enables the whole-image fallback in Recognize: when the
	// detector finds no cards (it is trained on photos of physical cards and
	// scores flat scans and official renders near zero) and the image's
	// height/width ratio is within this relative tolerance of CardH/CardW,
	// the whole image is embedded as one upright card. Keep it tight enough
	// to exclude common photo ratios (4:3, 16:9); 0 disables the fallback.
	FlatAspectTol float32
}

// YuGiOh returns the profile for Yu-Gi-Oh! cards (59x86mm, standard frame
// artwork window). The artwork fractions intentionally cover the common
// window of both standard and pendulum frames.
func YuGiOh() Profile {
	return Profile{
		DetectSize:        640,
		ConfThreshold:     0.25,
		IoUThreshold:      0.45,
		MinConf:           0.5,
		MinAreaFrac:       0.02,
		CardW:             472,
		CardH:             688,
		ArtX:              0.15,
		ArtY:              0.22,
		ArtW:              0.70,
		ArtH:              0.46,
		MaxPhotoSide:      1600,
		FlipRetryBelowSim: 0.82,
		FlatAspectTol:     0.05,
	}
}
