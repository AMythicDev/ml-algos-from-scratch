# notebook-image-extractor: Extract base64 encoded images from Jupyter Notebook files
# This script extracts all the images in a jupyter notebook file and saves them
# into individual .png files.
#
# This is useful in lab09 and lab10 where the models were trained on
# Google Collab where it become tedious for me to download all the images from 
# the ephemeral filesystem of the runtime.
#
# Usage
# 1. Navigate to the experiment folder.
# 2. Run command:
#       uv run ../notebook-image-extractor lab<N>.ipynb
# For non-uv users the command is 
#       python ../notebook-image-extractor lab<N>.ipynb

import base64
import sys
import json

nb_fn = sys.argv[1]

def fig_id_generator():
    i = 1
    while True:
        yield i
        i += 1
fig_id = fig_id_generator()

with open(nb_fn) as nb:
    cells = json.load(nb)["cells"]
    for cell in cells:
        if "outputs" in cell and len(cell["outputs"]) > 0:
            for output in cell["outputs"]:
                if "data" in output and "image/png" in output["data"]:
                    img_b64 = output["data"]["image/png"]
                    img = base64.b64decode(img_b64)
                    with open(f"fig{next(fig_id)}.png", "wb") as image_file:
                        image_file.write(img)
                        image_file.flush()
