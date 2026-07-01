from src.jpeg_utils import load_jpeg_as_ycbcr, inspect_jpeg

if __name__ == "__main__":
    inspect_jpeg("chilling.jpg")

    ycbcr = load_jpeg_as_ycbcr("chilling.jpg")
    print(f"YCbCr shape: {ycbcr.shape}, dtype: {ycbcr.dtype}")
    print(f"Y channel range: {ycbcr[:,:,0].min()} - {ycbcr[:,:,0].max()}")