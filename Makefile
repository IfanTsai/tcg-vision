ORT_VERSION := 1.23.2
ORT_DIR := third_party/onnxruntime-linux-x64-$(ORT_VERSION)

.PHONY: ort models test build cli

# Download the ONNX Runtime shared library.
ort:
	mkdir -p third_party
	curl -sL https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/onnxruntime-linux-x64-$(ORT_VERSION).tgz | tar -xz -C third_party
	@echo "libonnxruntime.so at $(ORT_DIR)/lib/"

# Export detector.onnx + embedder.onnx into models/ (needs Python; see export/).
models:
	mkdir -p models
	cd export && python export.py ../models

build:
	go build ./...

# Build the CLI binary at the repo root.
cli:
	go build -o tcgvision ./cmd/tcgvision

test:
	go test ./...
