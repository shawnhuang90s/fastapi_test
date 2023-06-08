# -*- coding: utf-8 -*-
# @Time: 2023/6/8 10:12
import io
from PIL import Image


def image2bytes(image, format="jpeg"):
    img_bytes = io.BytesIO()
    image = image.convert("RGB")
    image.save(img_bytes, format=format)
    image_bytes = img_bytes.getvalue()
    return image_bytes


def bytes2image(bytes_data):
    image = Image.open(io.BytesIO(bytes_data))
    return image


if __name__ == "__main__":
    img_path = "人生大事.jpg"
    image = Image.open(img_path)
    bytes_data = image2bytes(image)
    print(bytes_data)
    image_obj = bytes2image(bytes_data)
    print(image_obj)
