import matplotlib.pyplot as plt
from src.utils import get_image_var
from src.main_pipeline import get_marker_mask,get_inpainting_result,remove_background,tissue_segregation



def run_Vispro(imagepath):
    img_var, img_np = get_image_var(imagepath)
    marker_mask = get_marker_mask(img_var)
    inpainted_image = get_inpainting_result(img_np,marker_mask)
    tissue_rgba,tissue_rgb,tissue_mask= remove_background(inpainted_image,model_name='u2net')
    segregation_image, segment_mask = tissue_segregation(tissue_rgb,tissue_mask,tissue_value=20)

    f,a = plt.subplots(2,3)
    a[0,0].imshow(img_np)
    a[0,1].imshow(inpainted_image)
    a[0,2].imshow(tissue_rgba)
    a[1,0].imshow(tissue_mask)
    a[1,1].imshow(segment_mask)
    a[1,2].imshow(segregation_image)
    plt.show()

test_image_path = '/home/huifang/workspace/code/fiducial_remover/location_annotation/28.png'
# test_image_path = './test_data/tissue_hires_image.png'
run_Vispro(test_image_path)