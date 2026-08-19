// Command tcgvision is a CLI for the tcg-vision pipeline: build a reference
// index from a directory of card images, or recognize cards in a photo.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	tcgvision "github.com/IfanTsai/tcg-vision"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: tcgvision <index|recognize> [flags]")
		os.Exit(2)
	}

	var err error
	switch os.Args[1] {
	case "index":
		err = runIndex(os.Args[2:])
	case "recognize":
		err = runRecognize(os.Args[2:])
	default:
		err = fmt.Errorf("unknown subcommand %q", os.Args[1])
	}
	if err != nil {
		log.Fatal(err)
	}
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	img, _, err := image.Decode(f)

	return img, err
}

// runIndex embeds every image under -images (key = path relative to it) into
// an index file, resuming from -out if it already exists.
func runIndex(args []string) error {
	fs2 := flag.NewFlagSet("index", flag.ExitOnError)
	ortLib := fs2.String("ort", "", "path to libonnxruntime.so")
	embedder := fs2.String("embedder", "", "path to embedder.onnx")
	images := fs2.String("images", "", "root directory of reference images")
	out := fs2.String("out", "index.bin", "output index file")
	threads := fs2.Int("threads", runtime.NumCPU(), "intra-op threads")
	batch := fs2.Int("batch", 64, "embedding batch size")
	_ = fs2.Parse(args)

	pipe, err := tcgvision.New(tcgvision.Config{ORTLibPath: *ortLib, EmbedderPath: *embedder, Threads: *threads})
	if err != nil {
		return err
	}
	defer func() { _ = pipe.Close() }()

	idx := tcgvision.NewIndex(tcgvision.EmbedDim)
	if st, err := os.Stat(*out); err == nil && st.Size() > 0 {
		idx, err = tcgvision.LoadIndex(*out)
		if err != nil {
			return fmt.Errorf("resume from %s: %w", *out, err)
		}
		log.Printf("resuming: %d embeddings already in %s", idx.Len(), *out)
	}

	var files []string
	err = filepath.WalkDir(*images, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" {
			key := filepath.ToSlash(strings.TrimPrefix(path, strings.TrimSuffix(*images, "/")+"/"))
			if !idx.Has(key) {
				files = append(files, path)
			}
		}

		return nil
	})
	if err != nil {
		return fmt.Errorf("walk %s: %w", *images, err)
	}
	log.Printf("%d images to embed", len(files))

	root := strings.TrimSuffix(*images, "/") + "/"
	start := time.Now()
	done := 0
	for base := 0; base < len(files); base += *batch {
		end := min(base+*batch, len(files))
		chunk := files[base:end]

		imgs := make([]image.Image, len(chunk))
		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.NumCPU())
		for i, path := range chunk {
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-sem }()
				img, err := loadImage(path)
				if err != nil {
					log.Printf("skip %s: %v", path, err)

					return
				}
				imgs[i] = img
			}()
		}
		wg.Wait()

		for i, img := range imgs {
			if img == nil {
				continue
			}
			vec, err := pipe.EmbedReference(img)
			if err != nil {
				return fmt.Errorf("embed %s: %w", chunk[i], err)
			}
			if err := idx.Add(filepath.ToSlash(strings.TrimPrefix(chunk[i], root)), vec); err != nil {
				return err
			}
			done++
		}

		if (base / *batch)%20 == 0 || end == len(files) {
			elapsed := time.Since(start).Round(time.Second)
			log.Printf("%d/%d embedded (%s)", done, len(files), elapsed)
			if err := idx.Save(*out); err != nil {
				return err
			}
		}
	}

	if err := idx.Save(*out); err != nil {
		return err
	}
	log.Printf("saved %d embeddings to %s", idx.Len(), *out)

	return nil
}

// runRecognize prints detections + top-K matches for one photo as JSON.
func runRecognize(args []string) error {
	fs2 := flag.NewFlagSet("recognize", flag.ExitOnError)
	ortLib := fs2.String("ort", "", "path to libonnxruntime.so")
	detector := fs2.String("detector", "", "path to detector.onnx")
	embedder := fs2.String("embedder", "", "path to embedder.onnx")
	indexPath := fs2.String("index", "", "path to index file")
	photo := fs2.String("photo", "", "photo to recognize")
	topK := fs2.Int("k", 5, "matches per card")
	threads := fs2.Int("threads", 2, "intra-op threads")
	_ = fs2.Parse(args)

	pipe, err := tcgvision.New(tcgvision.Config{
		ORTLibPath: *ortLib, DetectorPath: *detector, EmbedderPath: *embedder, Threads: *threads,
	})
	if err != nil {
		return err
	}
	defer func() { _ = pipe.Close() }()

	idx, err := tcgvision.LoadIndex(*indexPath)
	if err != nil {
		return err
	}

	img, err := loadImage(*photo)
	if err != nil {
		return err
	}

	start := time.Now()
	dets, err := pipe.Recognize(img, idx, *topK)
	if err != nil {
		return err
	}
	log.Printf("%d cards in %s", len(dets), time.Since(start).Round(time.Millisecond))

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", " ")

	return enc.Encode(dets)
}
