package tcgvision

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// indexMagic identifies the on-disk index format (version suffix included).
const indexMagic = "TCGVIDX1"

// Match is one retrieval hit: the reference image key and cosine similarity.
type Match struct {
	Key string
	Sim float32
}

// Index is an in-memory reference-embedding store with brute-force cosine
// search. Vectors are stored L2-normalized as float32; the file format keeps
// them as float16 to halve size. Not safe for concurrent mutation; concurrent
// Search is fine.
type Index struct {
	dim    int
	keys   []string
	keyPos map[string]int
	vecs   []float32 // len(keys) * dim
}

// NewIndex creates an empty index for embeddings of the given dimension.
func NewIndex(dim int) *Index {
	return &Index{dim: dim, keyPos: make(map[string]int)}
}

// LoadIndex reads an index written by Save.
func LoadIndex(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open index: %w", err)
	}
	defer func() { _ = f.Close() }()

	r := bufio.NewReaderSize(f, 1<<20)
	magic := make([]byte, len(indexMagic))
	if _, err := readFull(r, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic) != indexMagic {
		return nil, fmt.Errorf("not a tcg-vision index (magic %q)", magic)
	}

	var dim, count uint32
	if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
		return nil, fmt.Errorf("read dim: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read count: %w", err)
	}
	if dim == 0 || dim > 1<<16 {
		return nil, fmt.Errorf("implausible dimension %d", dim)
	}

	ix := &Index{
		dim:    int(dim),
		keys:   make([]string, 0, count),
		keyPos: make(map[string]int, count),
		vecs:   make([]float32, 0, int(count)*int(dim)),
	}
	keyBuf := make([]byte, 1<<10)
	half := make([]uint16, dim)
	for range count {
		var keyLen uint16
		if err := binary.Read(r, binary.LittleEndian, &keyLen); err != nil {
			return nil, fmt.Errorf("read key len: %w", err)
		}
		if int(keyLen) > len(keyBuf) {
			keyBuf = make([]byte, keyLen)
		}
		if _, err := readFull(r, keyBuf[:keyLen]); err != nil {
			return nil, fmt.Errorf("read key: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, half); err != nil {
			return nil, fmt.Errorf("read vector: %w", err)
		}

		key := string(keyBuf[:keyLen])
		ix.keyPos[key] = len(ix.keys)
		ix.keys = append(ix.keys, key)
		for _, hv := range half {
			ix.vecs = append(ix.vecs, f16ToFloat32(hv))
		}
	}

	return ix, nil
}

// Dim returns the embedding dimension.
func (ix *Index) Dim() int { return ix.dim }

// Len returns the number of stored embeddings.
func (ix *Index) Len() int { return len(ix.keys) }

// Has reports whether key is present.
func (ix *Index) Has(key string) bool {
	_, ok := ix.keyPos[key]

	return ok
}

// Keys returns all stored keys (shared slice; do not mutate).
func (ix *Index) Keys() []string { return ix.keys }

// Add inserts or replaces the embedding for key. The vector must have the
// index dimension and be L2-normalized by the caller.
func (ix *Index) Add(key string, vec []float32) error {
	if len(vec) != ix.dim {
		return fmt.Errorf("dimension mismatch: got %d want %d", len(vec), ix.dim)
	}

	if pos, ok := ix.keyPos[key]; ok {
		copy(ix.vecs[pos*ix.dim:], vec)

		return nil
	}

	ix.keyPos[key] = len(ix.keys)
	ix.keys = append(ix.keys, key)
	ix.vecs = append(ix.vecs, vec...)

	return nil
}

// Rename moves the embedding stored under oldKey to newKey without touching
// the vector. It fails when oldKey is absent or newKey is already present.
func (ix *Index) Rename(oldKey, newKey string) error {
	pos, ok := ix.keyPos[oldKey]
	if !ok {
		return fmt.Errorf("key %q not found", oldKey)
	}

	if _, exists := ix.keyPos[newKey]; exists {
		return fmt.Errorf("key %q already exists", newKey)
	}

	delete(ix.keyPos, oldKey)
	ix.keyPos[newKey] = pos
	ix.keys[pos] = newKey

	return nil
}

// Search returns the topK most similar keys by dot product (== cosine for
// normalized vectors), descending.
func (ix *Index) Search(vec []float32, topK int) []Match {
	if len(vec) != ix.dim || topK <= 0 || len(ix.keys) == 0 {
		return nil
	}
	if topK > len(ix.keys) {
		topK = len(ix.keys)
	}

	sims := make([]float32, len(ix.keys))
	for i := range ix.keys {
		base := i * ix.dim
		var s float32
		for j, v := range vec {
			s += v * ix.vecs[base+j]
		}
		sims[i] = s
	}

	order := make([]int, len(sims))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return sims[order[a]] > sims[order[b]] })

	out := make([]Match, topK)
	for i := range topK {
		out[i] = Match{Key: ix.keys[order[i]], Sim: sims[order[i]]}
	}

	return out
}

// Save writes the index atomically (temp file + rename) as
// magic | u32 dim | u32 count | count x { u16 keyLen | key | dim x f16 }.
func (ix *Index) Save(path string) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tcgvidx-*")
	if err != nil {
		return fmt.Errorf("create temp index: %w", err)
	}
	defer func() { _ = os.Remove(tmp.Name()) }()

	w := bufio.NewWriterSize(tmp, 1<<20)
	if _, err := w.WriteString(indexMagic); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(ix.dim)); err != nil {
		return fmt.Errorf("write dim: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(ix.keys))); err != nil {
		return fmt.Errorf("write count: %w", err)
	}

	half := make([]uint16, ix.dim)
	for i, key := range ix.keys {
		if err := binary.Write(w, binary.LittleEndian, uint16(len(key))); err != nil {
			return fmt.Errorf("write key len: %w", err)
		}
		if _, err := w.WriteString(key); err != nil {
			return fmt.Errorf("write key: %w", err)
		}
		base := i * ix.dim
		for j := range ix.dim {
			half[j] = float32ToF16(ix.vecs[base+j])
		}
		if err := binary.Write(w, binary.LittleEndian, half); err != nil {
			return fmt.Errorf("write vector: %w", err)
		}
	}

	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush index: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temp index: %w", err)
	}

	if err := os.Rename(tmp.Name(), path); err != nil {
		return fmt.Errorf("rename index into place: %w", err)
	}

	return nil
}

func readFull(r *bufio.Reader, buf []byte) (int, error) {
	n := 0
	for n < len(buf) {
		m, err := r.Read(buf[n:])
		n += m
		if err != nil {
			return n, err
		}
	}

	return n, nil
}
