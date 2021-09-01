import os

## This file collects new vs old (Luca Colomba) paths for datasets and neural networks
# Notation: _dir = suffix for directories, _path = suffix for files
data_dir = '../data/'
geospatialdownloader_dir = '../data/geospatialdownloader'

# Neural netorks and outputs
test_train_dir           = os.path.join(data_dir, 'multimodal/test_train') #'/mnt/data2/colomba_data/multimodal/test_train'
pretrained_dir  = os.path.join(data_dir, 'pretrained_weights/') #'/mnt/data2/colomba_data/multimodal/test_selected_indexes'
unet_results_dir         = os.path.join(data_dir, 'multimodal/s2_burned_area_index_only_cut') #'/mnt/data2/colomba_data/multimodal/s2_burned_area_index_only_cut'
# base_result_dir          = os.path.join(data_dir, 'multimodal/base_result') #'/mnt/data2/colomba_data/multimodal_paper_v6'
base_result_dir          = os.path.join(data_dir, 'model_evaluations') #'/mnt/data2/colomba_data/multimodal_paper_v6'
final_double_unet_output = os.path.join(pretrained_dir, 'test_selected_indexes.pt')


# Datasets
# sentinel_hub_selected_dir = os.path.join(data_dir, 'sentinel-hub-selected') #'/mnt/data2/sentinel-hub-selected'
sentinel_hub_selected_dir = os.path.join(data_dir, 'sentinel-hub') #'/mnt/data2/sentinel-hub-selected'
sentinel_hub_correct_filtered_dir = os.path.join(data_dir, 'sentinel-hub-correct-filtered') #'/mnt/data2/sentinel-hub-correct-filtered'
satellite_csv_path = os.path.join(geospatialdownloader_dir, 'scripts/satellite_data.csv') #'/home/colomba/geospatialdownloader/scripts/satellite_data.CSV'
satellite_folds_csv_path = './dset_folds2.csv' # K-fold subdivision of Sentinel-2 dataset
# satellite_folds_csv_path = './dset_folds.csv' # K-fold subdivision of Sentinel-2 dataset


model_names=['Concatenated U-Net', 'PSP Net', 'Nested U-Net', 'Concatenated Nested U-Net', 'Seg-U-Net', 'Concatenated Seg-U-Net', 'Mixed', 'U-Net']
models = {
        0: "concat_unet",
        1: "pspnet",
        2: "nested_unet",
        3: "concat_nest_unet",
        4: "segnet",
        5: "concat_segnet",
        6: "concat_mixed",
        7: "unet"
    }

mask_intervals = [(0, 32), (33, 96), (97, 160), (161, 224), (225, 255)]
# Test prefix
test_prefix = ['EMSR221', 'EMSR371'] # Blue fold
# excluding associated validation set
# ignore_list = ["EMSR216_02TORREPEDRO_02GRADING_MAP_v1_vector","EMSR216_04RALA_02GRADING_MAP_v1_vector","EMSR216_05ELCALAR_02GRADING_MAP_v2_vector","EMSR248_01PINODELORO_02GRADING_MAP_v1_vector","EMSR248_03MEDINILLA_02GRADING_MAP_v1_vector","EMSR248_04HOYOSDEMIGUELMUNOZ_02GRADING_MAP_v1_vector","EMSR248_05HOYOCASERO_02GRADING_MAP_v1_vector","EMSR250_01MARINHAGRANDE_02GRADING_MAP_v2_vector","EMSR290_03MANSBO_02GRADING_MAP_v1_vector","EMSR298_02HAMMARSTRAND_02GRADING_MAP_v1_vector","EMSR298_06GROTINGEN_02GRADING_MAP_v1_vector","EMSR302_01NERVA_02GRADING_MAP_v1_vector","EMSR302_06NERVADETAIL_02GRADING_MAP_v1_vector","EMSR302_07ELPERALEJO_02GRADING_MAP_v1_vector","EMSR365_AOI01_GRA_PRODUCT_r1_RTP01_v1_vector","EMSR368_AOI01_GRA_PRODUCT_r1_RTP01_v3_vector","EMSR371_AOI01_GRA_PRODUCT_r1_RTP01_v2_vector","EMSR372_AOI04_GRA_PRODUCT_r1_RTP01_v3_vector","EMSR373_AOI01_GRA_PRODUCT_r1_RTP01_v2_vector"]
ignore_list = []#'EMSR368_AOI01_GRA_PRODUCT_r1_RTP01_v3_vector', 'EMSR248_04HOYOSDEMIGUELMUNOZ_02GRADING_MAP_v1_vector', 'EMSR248_05HOYOCASERO_02GRADING_MAP_v1_vector', 'EMSR248_01PINODELORO_02GRADING_MAP_v1_vector', 'EMSR248_03MEDINILLA_02GRADING_MAP_v1_vector']
prefix_list = {"EMSR216", "EMSR216", "EMSR216", "EMSR216", "EMSR248", "EMSR248", "EMSR248", "EMSR248", "EMSR250",
               "EMSR290", "EMSR298", "EMSR298", "EMSR302", "EMSR302", "EMSR302", "EMSR365", "EMSR368", "EMSR371",
               "EMSR372", "EMSR373"}
