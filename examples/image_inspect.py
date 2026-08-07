from stego.img_utils import load_jpeg_as_ycbcr, inspect_jpeg
from stego.constants import IMAGE_PATH


if __name__ == "__main__":

    inspect_jpeg(str(IMAGE_PATH))

    ycbcr = load_jpeg_as_ycbcr(str(IMAGE_PATH))
    print(f"YCbCr shape: {ycbcr.shape}, dtype: {ycbcr.dtype}")
    print(f"Y channel range: {ycbcr[:,:,0].min()} - {ycbcr[:,:,0].max()}")