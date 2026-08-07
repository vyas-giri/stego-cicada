# Stego

Stego is a Python steganography project for hiding encrypted messages inside images and analyzing them afterward.

It currently supports two embedding pipelines:

- PNG spatial LSB embedding for lossless image workflows
- JPEG DCT coefficient embedding using `jpegio` for compressed-domain hiding

Messages are first encrypted with password-based AES-GCM, then protected with Reed-Solomon coding before being embedded into the carrier image.

## Features

- Password-based encryption using AES-GCM
- Reed-Solomon error correction for payload resilience
- PNG spatial LSB embedding
- JPEG compressed-domain DCT embedding
- Chi-square steganalysis for PNG and JPEG images
- CLI for hiding, extracting, and analyzing images
- Test coverage for crypto, PNG pipeline, JPEG pipeline, analysis, and CLI workflows

## Project Structure

- `stego/constants.py` - project paths and payload magic header
- `stego/crypto.py` - key derivation and AES-GCM encryption/decryption
- `stego/encoder.py` - payload encoding and embedding helpers
- `stego/decoder.py` - payload extraction and decoding helpers
- `stego/img_utils.py` - image load/save utilities
- `stego/steganography.py` - high-level hide/extract orchestration
- `stego/steganalysis.py` - chi-square analysis helpers
- `stego/main.py` - command-line entry point
- `examples/image_inspect.py` - example script for inspecting an image
- `tests/test_stego.py` - end-to-end and unit tests
- `tests/test_detectionRates.py` - detection-rate benchmark for JPEG DCT analysis

## Installation

Create and activate a virtual environment, then install the project in editable mode:

```bash
python -m pip install -e .
```

If you are working on Linux and `jpegio` needs native dependencies, install build tools first if necessary.

## Usage

Hide a message:

```bash
python -m stego.main hide data/input_imgs/chilling.jpg "Secret message" -p "mypassword" -m auto
```

Extract a message:

```bash
python -m stego.main extract data/output_imgs/stego_example.png -p "mypassword" -m auto
```

Analyze an image:

```bash
python -m stego.main analyze data/output_imgs/stego_example.png -m auto
```

You can also force the analysis method:

```bash
python -m stego.main analyze data/output_imgs/stego_example.png -m png_lsb
python -m stego.main analyze data/output_imgs/stego_example.jpg -m jpeg_dct
```

## Methods

### `png_lsb`

- Uses spatial LSB embedding on PNG images
- Best for lossless round-trip behavior
- Recommended for quick, stable tests and general hiding workflows

### `jpeg_dct`

- Uses quantized DCT coefficient embedding
- Designed for JPEG files
- Keeps the work in the compressed domain instead of re-encoding through pixels

### `auto`

- Chooses the method based on the input file extension
- `.png` uses `png_lsb`
- `.jpg` and `.jpeg` use `jpeg_dct`

## Steganalysis

The project includes a simple chi-square steganalysis command that estimates whether an image likely contains hidden data.

- `analyze_jpeg_dct()` checks AC coefficient pair statistics in JPEGs
- `analyze_png_lsb()` checks even/odd pixel value distributions in PNGs
- `analyze_image()` chooses the correct analyzer based on file type or an explicit method

This is useful for comparing how detectable different embedding strategies are as payload size increases.

## Example

Inspect an image with the example script:

```bash
python examples/image_inspect.py
```

## Testing

Run the main test suite:

```bash
python tests/test_stego.py
```

Run the steganalysis benchmark:

```bash
python tests/test_detectionRates.py
```

Or, if you are using pytest directly:

```bash
python -m pytest -q
```

## Notes

- The payload format includes a magic header and payload length before the encrypted data.
- Internal modules are packaged under `stego`, so imports should use `from stego...`.
- The project supports both a lossless spatial pipeline and a compressed-domain JPEG pipeline.
- The `analyze` CLI command uses the same package-level steganalysis helpers as the benchmark.

## Future Work

- Add more image formats such as BMP and TIFF
- Improve capacity reporting and validation
- Add stronger CLI output and error handling
- Expand test coverage for corrupted payloads and wrong passwords
- Add quality and robustness metrics for different embedding methods
