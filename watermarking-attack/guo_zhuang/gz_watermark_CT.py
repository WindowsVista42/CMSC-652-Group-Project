import os
import sys
import time
from glob import glob
from pathlib import Path

# import hashlib
# from cryptography.hazmat.primitives.asymmetric import rsa, padding
# from cryptography.hazmat.primitives import hashes
# from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
# from cryptography.hazmat.backends import default_backend

from PIL import Image
# from io import BytesIO
# from typing import List, Tuple

from watermarker import MedicalImageWatermarker

if __name__ == "__main__":

	print("Starting watermarker")

	watermarker = MedicalImageWatermarker()

	private_key, public_key = watermarker.generate_keys()

	#Get list of all images from ct scan

	ct_raw_dir = '../datasets/ctscan/raw/'

	ct_gz_dir = '../datasets/ctscan/guo_zhuang/'
	os.makedirs(os.path.join(ct_gz_dir, 'COVID/'), exist_ok = True)
	os.makedirs(os.path.join(ct_gz_dir, 'non-COVID/'), exist_ok = True)

	# print(os.listdir(os.path.join(ct_raw_dir, 'COVID/')))

	i = 0
	for imfile in os.listdir(os.path.join(ct_raw_dir, 'COVID/')):
		imfile = glob(os.path.join(ct_raw_dir, 'COVID/', imfile))

		assert len(imfile) == 1
		"More than one or zero files found"

		# watermarked_img = watermarker.embed_watermark(os.path.join(ct_raw_dir, 'COVID/', imfile[0]), private_key,
		# 	"a", [(10, 10), (20, 10), (20, 20), (10, 20)])
		watermarked_img = watermarker.embed_watermark(imfile[0], private_key,
			"a", [(10, 10), (20, 10), (20, 20), (10, 20)])

		watermarked_img.save(os.path.join(ct_gz_dir, 'COVID/', imfile[0].split('/')[-1]))
		i+=1
		if i % 50 == 0:
			print(i)

	for imfile in os.listdir(os.path.join(ct_raw_dir, 'non-COVID/')):
		imfile = glob(os.path.join(ct_raw_dir, 'non-COVID/', imfile))

		assert len(imfile) == 1
		"More than one or zero files found"

		watermarked_img = watermarker.embed_watermark(imfile[0], private_key,
			"a", [(10, 10), (20, 10), (20, 20), (10, 20)])

		watermarked_img.save(os.path.join(ct_gz_dir, 'non-COVID/', imfile[0].split('/')[-1]))
		i+=1
		if i % 50 == 0:
			print(i)
