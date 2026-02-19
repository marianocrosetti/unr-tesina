from datasets import sigcomp2009
from visualization import preprocess_choose
import torchvision
import torchvision.transforms.functional as TF


def image_binarization():
    ds = sigcomp2009()
    timg = torchvision.io.read_image(ds[0][0])
    img = TF.to_pil_image(timg)
    preprocess_choose(img).savefig('./media/image_binarization.png')


if __name__ == "__main__":
    image_binarization()

